"""corerec-bot logic, offline: checklist, labels, junit parsing, comment."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "bot", Path(__file__).resolve().parents[1] / ".github" / "corerec_bot" / "bot.py")
bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bot)


def test_labels_follow_paths():
    assert bot.labels_for(["corerec/engines/sar.py", "docs/x.md", "README.md"]) == ["documentation", "engines"]


def test_code_without_test_or_changelog_blocks_approval():
    rows = bot.checklist("Fixes #12. SAR now accepts the triple; tested locally.", ["corerec/engines/sar.py"], 20)
    assert not bot.checklist_ok(rows)
    rows = bot.checklist("Fixes #12. SAR now accepts the triple; tested locally.",
                         ["corerec/engines/sar.py", "tests/test_sar.py", "CHANGELOG.md"], 20)
    assert bot.checklist_ok(rows)


def test_large_pr_is_info_only():
    rows = bot.checklist("Fixes #1, a long enough description here.", ["docs/a.md"], 5000)
    assert bot.checklist_ok(rows)


def test_junit_failures_and_comment():
    xml = ('<testsuites><testsuite><testcase classname="tests.test_a" name="test_ok"/>'
           '<testcase classname="tests.test_a" name="test_bad"><failure/></testcase></testsuite></testsuites>')
    tests = bot.parse_junit({"Python 3.11": xml})
    assert tests["Python 3.11"] == {"tests": 2, "failed": ["tests/test_a::test_bad"], "skipped": 0}
    body = bot.render(checks=[("Description", True, "")], tests=tests, ci="failure", sha="abcdef123")
    assert bot.MARK in body and "1 passed, 1 failed, 0 skipped" in body and "test_bad" in body
    assert bot.read_state(body) == {"checks_ok": True}


class FakeGH:
    """Records writes; serves canned reads."""

    def __init__(self, pr, reviews=(), comments=()):
        self.pr, self.reviews, self.comments, self.writes = pr, list(reviews), list(comments), []

    def call(self, method, path, data=None):
        self.writes.append((method, path, data))
        return {}

    def paged(self, path):
        if path.startswith("/pulls?state=open"):
            return [self.pr]
        if path.endswith("/files"):
            return [{"filename": f} for f in ("corerec/engines/sar.py", "tests/test_sar.py", "CHANGELOG.md")]
        if path.endswith("/reviews"):
            return self.reviews
        if path.endswith("/comments"):
            return self.comments
        return []

    bot_comment = bot.GitHub.bot_comment
    upsert = bot.GitHub.upsert
    ensure_labels = bot.GitHub.ensure_labels


def _pr(sha="abc123"):
    return {"number": 7, "body": "Fixes #12. SAR accepts the triple; tested with the contract suite.",
            "additions": 30, "deletions": 2, "draft": False, "head": {"sha": sha},
            "user": {"login": "someone"}, "author_association": "CONTRIBUTOR"}


def _run(conclusion):
    return {"workflow_run": {"head_sha": "abc123", "conclusion": conclusion}}


def test_report_approves_when_green():
    gh = FakeGH(_pr())
    bot.report(gh, _run("success"), "")
    assert any(m == "POST" and p == "/pulls/7/reviews" and d["event"] == "APPROVE" for m, p, d in gh.writes)


def test_report_never_approves_a_failure():
    gh = FakeGH(_pr())
    bot.report(gh, _run("failure"), "")
    assert not any(p == "/pulls/7/reviews" for _, p, _ in gh.writes)


def test_report_does_not_approve_twice():
    gh = FakeGH(_pr(), reviews=[{"user": {"login": bot.BOT_LOGIN}, "state": "APPROVED", "commit_id": "abc123"}])
    bot.report(gh, _run("success"), "")
    assert not any(p == "/pulls/7/reviews" for _, p, _ in gh.writes)


def test_new_commits_dismiss_an_old_approval():
    old = {"id": 5, "user": {"login": bot.BOT_LOGIN}, "state": "APPROVED", "commit_id": "old"}
    gh = FakeGH(_pr("new"), reviews=[old])
    bot.triage(gh, {"action": "synchronize", "pull_request": _pr("new")})
    assert ("PUT", "/pulls/7/reviews/5/dismissals") in [(m, p) for m, p, _ in gh.writes]


def test_comment_is_updated_in_place():
    gh = FakeGH(_pr(), comments=[{"id": 9, "body": bot.MARK + " old"}])
    bot.triage(gh, {"action": "edited", "pull_request": _pr()})
    assert [(m, p) for m, p, _ in gh.writes if "comments" in p] == [("PATCH", "/issues/comments/9")]
