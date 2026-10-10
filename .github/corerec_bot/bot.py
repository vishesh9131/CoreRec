"""
corerec-bot: checks every pull request and keeps one comment on it up to date.

Two entry points, one per workflow (.github/workflows/corerec-bot-*.yml):

    python bot.py triage    # pull_request_target: checklist, labels, welcome
    python bot.py report    # workflow_run after "CoreRec CI": tests, approval

`triage` only reads PR metadata through the API and never runs the PR's code,
which is what makes it safe to hand it the bot's token on fork PRs. `report`
reads test results that CI uploaded as an artifact.

Pure logic (checklist, labels, junit parsing, comment rendering) is kept
separate from the API calls so tests/test_corerec_bot.py can check it offline.
Stdlib only.
"""

import json
import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

MARK = "<!-- corerec-bot -->"
BOT_LOGIN = os.environ.get("COREREC_BOT_LOGIN", "corerec-bot[bot]")

# path prefix -> label; first match per file, a PR gets the union
AREAS = [
    ("corerec/engines/", "engines"),
    ("corerec/nn/", "nn"),
    ("corerec/serving/", "serving"),
    ("corerec/evaluation/", "evaluation"),
    ("corerec/export.py", "export"),
    ("corerec/pipelines/", "pipelines"),
    ("corerec/retrieval/", "pipelines"),
    ("corerec/ranking/", "pipelines"),
    ("corerec/reranking/", "pipelines"),
    ("docs/", "documentation"),
    ("tests/", "tests"),
    (".github/", "ci"),
]
LABEL_COLORS = {"engines": "1d76db", "nn": "5319e7", "serving": "0e8a16", "evaluation": "fbca04",
                "export": "c5def5", "pipelines": "bfdadc", "documentation": "0075ca",
                "tests": "d4c5f9", "ci": "000000", "first-time contributor": "7057ff"}
LARGE_PR = 800  # changed lines


# --------------------------------------------------------------------- logic
def labels_for(files: Iterable[str]) -> List[str]:
    out = set()
    for f in files:
        if f.endswith(".md") and not f.startswith(("corerec/", "tests/")):
            out.add("documentation")
            continue
        for prefix, label in AREAS:
            if f.startswith(prefix):
                out.add(label)
                break
    return sorted(out)


def checklist(body: Optional[str], files: Iterable[str], changed_lines: int) -> List[Tuple[str, bool, str]]:
    """(rule, passed, hint) per CONTRIBUTING.md. Rules marked info never block approval."""
    files = list(files)
    text = re.sub(r"<!--.*?-->", "", body or "", flags=re.S).strip()
    code = [f for f in files if f.startswith("corerec/") and f.endswith(".py")]
    rows = [
        ("Description", len(text) >= 30,
         "Say what was wrong, how you fixed it and how you tested it."),
        ("Linked issue", bool(re.search(r"#\d+", text)),
         "Reference the issue, e.g. `Fixes #123`."),
    ]
    if code:
        rows.append(("Tests", any(f.startswith("tests/") for f in files),
                     "Library code changed without a test. Add one that fails without the fix."))
        rows.append(("CHANGELOG", "CHANGELOG.md" in files,
                     "Add a line under `[Unreleased]` in CHANGELOG.md."))
    rows.append(("Size (info)", changed_lines <= LARGE_PR,
                 f"{changed_lines} changed lines; consider splitting into smaller PRs."))
    return rows


def checklist_ok(rows) -> bool:
    return all(ok for rule, ok, _ in rows if not rule.endswith("(info)"))


def parse_junit(xml_texts: Dict[str, str]) -> Dict[str, Dict]:
    """{name: junit xml} -> {name: {"tests", "failed": [test ids], "skipped"}}."""
    out = {}
    for name, text in xml_texts.items():
        root = ET.fromstring(text)
        failed, total, skipped = [], 0, 0
        for case in root.iter("testcase"):
            total += 1
            skipped += case.find("skipped") is not None
            if case.find("failure") is not None or case.find("error") is not None:
                cls = case.get("classname", "").replace(".", "/")
                failed.append(f"{cls}::{case.get('name')}")
        out[name] = {"tests": total, "failed": failed, "skipped": skipped}
    return out


def render(checks=None, tests=None, ci=None, sha=None, welcome=False, author="") -> str:
    """The single bot comment. Sections left as None keep a placeholder."""
    parts = [MARK, "### corerec-bot"]
    if welcome:
        parts.append(f"Thanks for your first contribution, @{author}! "
                     "[CONTRIBUTING.md](../blob/main/CONTRIBUTING.md) has the setup and the test "
                     "command CI runs. A maintainer reviews every PR before it's merged.")
    parts.append("**Checklist**")
    if checks is None:
        parts.append("_pending_")
    else:
        parts.append("\n".join(f"- [{'x' if ok else ' '}] **{rule}**" + ("" if ok else f": {hint}")
                               for rule, ok, hint in checks))
    parts.append("**Tests**")
    if tests is None and ci is None:
        parts.append("_waiting for CI_")
    else:
        if ci:
            parts.append(f"CI: **{ci}**" + (f" on `{sha[:7]}`" if sha else ""))
        failed = sorted({t for r in (tests or {}).values() for t in r["failed"]})
        if tests:
            parts.append("\n".join(
                f"- {name}: {r['tests'] - len(r['failed']) - r.get('skipped', 0)} passed, "
                f"{len(r['failed'])} failed, {r.get('skipped', 0)} skipped"
                for name, r in sorted(tests.items())))
        if failed:
            parts.append("<details><summary>Failing tests</summary>\n\n"
                         + "\n".join(f"- `{t}`" for t in failed[:50]) + "\n</details>")
    state = {"checks_ok": None if checks is None else checklist_ok(checks)}
    parts.append(f"<!-- state {json.dumps(state)} -->")
    return "\n\n".join(parts)


def read_state(comment_body: str) -> Dict:
    m = re.search(r"<!-- state (\{.*?\}) -->", comment_body or "")
    return json.loads(m.group(1)) if m else {}


def section(comment_body: str, title: str) -> Optional[str]:
    """Raw text of one section of an existing comment, to carry it over."""
    m = re.search(rf"\*\*{title}\*\*\n\n(.*?)(?=\n\n\*\*|\n\n<!-- state)", comment_body or "", re.S)
    return m.group(1) if m else None


# ----------------------------------------------------------------------- API
class GitHub:
    def __init__(self, repo: str, token: str, dry_run: bool = False):
        self.base, self.token, self.dry = f"https://api.github.com/repos/{repo}", token, dry_run

    def call(self, method: str, path: str, data=None):
        if self.dry and method != "GET":
            print(f"[dry-run] {method} {path} {json.dumps(data)[:300] if data else ''}")
            return {}
        req = urllib.request.Request(self.base + path, method=method,
                                     data=None if data is None else json.dumps(data).encode())
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        with urllib.request.urlopen(req) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}

    def paged(self, path: str):
        out, page = [], 1
        while True:
            sep = "&" if "?" in path else "?"
            got = self.call("GET", f"{path}{sep}per_page=100&page={page}")
            out += got
            if len(got) < 100:
                return out
            page += 1

    def bot_comment(self, pr: int):
        return next((c for c in self.paged(f"/issues/{pr}/comments") if MARK in (c["body"] or "")), None)

    def upsert(self, pr: int, body: str):
        c = self.bot_comment(pr)
        if c:
            self.call("PATCH", f"/issues/comments/{c['id']}", {"body": body})
        else:
            self.call("POST", f"/issues/{pr}/comments", {"body": body})

    def ensure_labels(self, names: List[str]):
        have = {lab["name"] for lab in self.paged("/labels")}
        for n in names:
            if n not in have:
                self.call("POST", "/labels", {"name": n, "color": LABEL_COLORS.get(n, "ededed")})


# ------------------------------------------------------------------ commands
def triage(gh: GitHub, event: Dict):
    pr = event["pull_request"]
    n = pr["number"]
    files = [f["filename"] for f in gh.paged(f"/pulls/{n}/files")]
    checks = checklist(pr.get("body"), files, pr["additions"] + pr["deletions"])
    first = pr.get("author_association") in ("FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER")
    labels = labels_for(files) + (["first-time contributor"] if first else [])
    gh.ensure_labels(labels)
    if labels:
        gh.call("POST", f"/issues/{n}/labels", {"labels": labels})

    old = gh.bot_comment(n)
    body = render(checks=checks, welcome=first, author=pr["user"]["login"])
    if old and event.get("action") != "synchronize":
        # keep the last test report unless new commits made it stale
        tests_txt = section(old["body"], "Tests")
        if tests_txt:
            body = body.replace("_waiting for CI_", tests_txt)
    gh.upsert(n, body)

    if event.get("action") == "synchronize":
        # new commits: an approval for an older commit no longer means anything
        for r in gh.paged(f"/pulls/{n}/reviews"):
            if (r["user"]["login"] == BOT_LOGIN and r["state"] == "APPROVED"
                    and r["commit_id"] != pr["head"]["sha"]):
                gh.call("PUT", f"/pulls/{n}/reviews/{r['id']}/dismissals",
                        {"message": "New commits pushed; corerec-bot will re-check."})


def report(gh: GitHub, event: Dict, results_dir: str):
    run = event["workflow_run"]
    sha = run["head_sha"]
    # workflow_run.pull_requests is empty for fork PRs, so match on the head commit
    pr = next((p for p in gh.paged("/pulls?state=open") if p["head"]["sha"] == sha), None)
    if pr is None:
        print(f"no open PR at {sha}; nothing to do")
        return
    n = pr["number"]
    xmls = {p.parent.name.replace("test-results-", "Python "): p.read_text()
            for p in Path(results_dir).glob("*/*.xml")} if results_dir else {}
    tests = parse_junit(xmls) if xmls else None
    conclusion = run["conclusion"]

    files = [f["filename"] for f in gh.paged(f"/pulls/{n}/files")]
    checks = checklist(pr.get("body"), files, pr["additions"] + pr["deletions"])
    first = pr.get("author_association") in ("FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER")
    gh.upsert(n, render(checks=checks, tests=tests, ci=conclusion, sha=sha,
                        welcome=first, author=pr["user"]["login"]))

    already = any(r["user"]["login"] == BOT_LOGIN and r["state"] == "APPROVED"
                  and r["commit_id"] == sha for r in gh.paged(f"/pulls/{n}/reviews"))
    if conclusion == "success" and checklist_ok(checks) and not pr["draft"] and not already:
        gh.call("POST", f"/pulls/{n}/reviews", {
            "commit_id": sha, "event": "APPROVE",
            "body": "CI is green and the checklist is complete. A maintainer still reviews and merges."})


def main(argv):
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    gh = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ.get("GH_TOKEN", ""),
                dry_run=os.environ.get("COREREC_BOT_DRY_RUN") == "1")
    if argv[1] == "triage":
        triage(gh, event)
    elif argv[1] == "report":
        report(gh, event, argv[2] if len(argv) > 2 else "")
    else:
        raise SystemExit("usage: bot.py triage|report [results_dir]")


if __name__ == "__main__":
    main(sys.argv)
