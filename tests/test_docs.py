"""
Documentation integrity tests for CoreRec.

Tests cover:
  1. MkDocs build (--strict mode)
  2. Sphinx build
  3. All nav-referenced files exist
  4. Sphinx toctree files exist
  5. Extra assets (CSS/JS/includes) exist
  6. Autodoc modules are importable
  7. Internal markdown links resolve
  8. Code blocks in docs have valid Python syntax
  9. No broken image references
"""

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
SOURCE_DIR = DOCS_DIR / "source"
MKDOCS_YML = ROOT / "mkdocs.yml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_nav_paths(mkdocs_yml: Path) -> list[str]:
    """Extract all .md paths from the mkdocs.yml nav section via regex."""
    text = mkdocs_yml.read_text()
    # Matches entries like "  - index.md" or "  - Installation: getting-started/installation.md"
    paths: list[str] = []
    for m in re.finditer(r":\s+(\S+\.md)", text):
        paths.append(m.group(1))
    for m in re.finditer(r"^\s+-\s+(\S+\.md)\s*$", text, re.MULTILINE):
        paths.append(m.group(1))
    return sorted(set(paths))


def _extract_toctree_entries(md_file: Path) -> list[str]:
    """Extract entries from Sphinx toctree directives in a markdown file.

    Toctree blocks look like:
        ```{toctree}
        ---
        hidden: true
        maxdepth: 2
        caption: Getting Started
        ---
        installation
        quickstart
        concepts
        ```
    Lines between the two ``---`` are YAML options (skipped).
    Lines after the second ``---`` are actual toctree entries.
    """
    text = md_file.read_text()
    entries: list[str] = []
    in_toctree = False
    dash_count = 0  # counts --- delimiters seen
    for line in text.splitlines():
        if "```{toctree}" in line:
            in_toctree = True
            dash_count = 0
            continue
        if in_toctree:
            if line.strip() == "```":
                in_toctree = False
                continue
            if line.strip() == "---":
                dash_count += 1
                continue
            # Only collect entries after both --- delimiters (past options)
            if dash_count >= 2 and line.strip():
                entries.append(line.strip())
    return entries


def _extract_automodule_paths(directory: Path) -> list[str]:
    """Extract all automodule/autoclass dotted paths from rst directives."""
    modules: list[str] = []
    for md_file in directory.rglob("*.md"):
        text = md_file.read_text()
        for m in re.finditer(r"\.\.\s+auto(?:module|class)::\s+([\w.]+)", text):
            modules.append(m.group(1))
    return sorted(set(modules))


def _extract_python_code_blocks(md_file: Path) -> list[tuple[int, str]]:
    """Extract python code blocks from a markdown file.

    Returns (line_number, code_string) pairs.
    """
    text = md_file.read_text()
    blocks: list[tuple[int, str]] = []
    pattern = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
    for m in pattern.finditer(text):
        line_no = text[: m.start()].count("\n") + 1
        blocks.append((line_no, m.group(1)))
    return blocks


def _extract_md_links(md_file: Path) -> list[tuple[int, str]]:
    """Extract markdown link targets from a file.

    Returns (line_number, link_target) for internal relative links.
    """
    links: list[tuple[int, str]] = []
    text = md_file.read_text()
    for i, line in enumerate(text.splitlines(), 1):
        for m in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", line):
            target = m.group(2)
            # Skip external links, anchors, and special sphinx refs
            if target.startswith(("http://", "https://", "#", "mailto:", "{ref}")):
                continue
            # Strip anchors from local links
            target = target.split("#")[0]
            if target:
                links.append((i, target))
    return links


# ===========================================================================
# Test 1: MkDocs nav file existence
# ===========================================================================

class TestMkDocsNavFiles:
    """Every file referenced in mkdocs.yml nav must exist under docs/."""

    nav_paths = _extract_nav_paths(MKDOCS_YML) if MKDOCS_YML.exists() else []

    @pytest.mark.parametrize("rel_path", nav_paths, ids=nav_paths)
    def test_nav_file_exists(self, rel_path: str):
        full = DOCS_DIR / rel_path
        assert full.exists(), (
            f"mkdocs.yml nav references '{rel_path}' but docs/{rel_path} does not exist"
        )


# ===========================================================================
# Test 2: MkDocs extra assets existence
# ===========================================================================

class TestMkDocsAssets:
    """Extra CSS, JS, and snippet files referenced by mkdocs.yml must exist."""

    def test_extra_css_exists(self):
        css_file = DOCS_DIR / "stylesheets" / "extra.css"
        assert css_file.exists(), f"Missing extra CSS: {css_file}"

    def test_mathjax_js_exists(self):
        js_file = DOCS_DIR / "javascripts" / "mathjax.js"
        assert js_file.exists(), f"Missing mathjax JS: {js_file}"

    def test_abbreviations_snippet_exists(self):
        snippet = DOCS_DIR / "includes" / "abbreviations.md"
        assert snippet.exists(), f"Missing abbreviations snippet: {snippet}"


# ===========================================================================
# Test 3: Sphinx source toctree entries exist
# ===========================================================================

class TestSphinxToctree:
    """Every toctree entry in source/index.md must resolve to a file."""

    toctree_entries = (
        _extract_toctree_entries(SOURCE_DIR / "index.md")
        if (SOURCE_DIR / "index.md").exists()
        else []
    )

    @pytest.mark.parametrize("entry", toctree_entries, ids=toctree_entries)
    def test_toctree_file_exists(self, entry: str):
        # Sphinx toctree entries are relative to the file's directory
        # and don't include .md extension sometimes
        candidates = [
            SOURCE_DIR / entry,
            SOURCE_DIR / f"{entry}.md",
            SOURCE_DIR / f"{entry}.rst",
            SOURCE_DIR / f"{entry}/index.md",
        ]
        found = any(c.exists() for c in candidates)
        assert found, (
            f"Sphinx toctree entry '{entry}' in source/index.md does not resolve to a file. "
            f"Checked: {[str(c) for c in candidates]}"
        )


# ===========================================================================
# Test 4: Autodoc modules are importable (with mocks)
# ===========================================================================

class TestAutodocModules:
    """Every module referenced by automodule/autoclass must be importable."""

    autodoc_modules = (
        _extract_automodule_paths(SOURCE_DIR)
        if SOURCE_DIR.exists()
        else []
    )

    @pytest.mark.parametrize("dotted_path", autodoc_modules, ids=autodoc_modules)
    def test_module_importable(self, dotted_path: str):
        # For autoclass directives, import the parent module
        parts = dotted_path.rsplit(".", 1)
        module_path = dotted_path

        # Try importing the full path first; if it fails, try parent
        try:
            importlib.import_module(module_path)
        except ImportError:
            if len(parts) == 2:
                try:
                    mod = importlib.import_module(parts[0])
                    assert hasattr(mod, parts[1]), (
                        f"Module '{parts[0]}' imported but has no attribute '{parts[1]}'"
                    )
                except ImportError as e:
                    pytest.fail(
                        f"autodoc references '{dotted_path}' but import failed: {e}"
                    )
            else:
                pytest.fail(
                    f"autodoc references '{dotted_path}' but import failed"
                )


# ===========================================================================
# Test 5: Python code blocks have valid syntax
# ===========================================================================

class TestCodeBlockSyntax:
    """Python code blocks in documentation must have valid syntax."""

    md_files = sorted(SOURCE_DIR.rglob("*.md")) if SOURCE_DIR.exists() else []

    @pytest.mark.parametrize(
        "md_file",
        md_files,
        ids=[str(f.relative_to(ROOT)) for f in md_files] if md_files else [],
    )
    def test_python_blocks_parse(self, md_file: Path):
        blocks = _extract_python_code_blocks(md_file)
        errors = []
        for line_no, code in blocks:
            try:
                compile(code, str(md_file), "exec")
            except SyntaxError as e:
                errors.append(
                    f"  Line ~{line_no}: SyntaxError: {e.msg} (line {e.lineno} in block)"
                )
        if errors:
            pytest.fail(
                f"Invalid Python syntax in {md_file.relative_to(ROOT)}:\n"
                + "\n".join(errors)
            )


# ===========================================================================
# Test 6: Internal markdown links resolve
# ===========================================================================

class TestInternalLinks:
    """Relative markdown links must point to existing files."""

    md_files = sorted(SOURCE_DIR.rglob("*.md")) if SOURCE_DIR.exists() else []

    @pytest.mark.parametrize(
        "md_file",
        md_files,
        ids=[str(f.relative_to(ROOT)) for f in md_files] if md_files else [],
    )
    def test_internal_links_resolve(self, md_file: Path):
        links = _extract_md_links(md_file)
        broken = []
        for line_no, target in links:
            # Resolve relative to the file's parent
            resolved = (md_file.parent / target).resolve()
            # Also try with .md appended
            resolved_md = (md_file.parent / f"{target}.md").resolve()
            if not resolved.exists() and not resolved_md.exists():
                broken.append(f"  Line {line_no}: '{target}' -> not found")
        if broken:
            pytest.fail(
                f"Broken links in {md_file.relative_to(ROOT)}:\n"
                + "\n".join(broken)
            )


# ===========================================================================
# Test 7: MkDocs build (--strict)
# ===========================================================================

class TestMkDocsBuild:
    """MkDocs must build without errors in --strict mode."""

    @pytest.mark.docs_build
    def test_mkdocs_build_strict(self):
        result = subprocess.run(
            [sys.executable, "-m", "mkdocs", "build", "--strict", "--clean"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            # Extract the most useful error lines
            stderr = result.stderr or ""
            stdout = result.stdout or ""
            output = stderr + stdout
            # Filter to warning/error lines
            important = [
                l for l in output.splitlines()
                if any(kw in l.lower() for kw in ("error", "warning", "not found", "missing"))
            ]
            msg = "\n".join(important[:30]) if important else output[-2000:]
            pytest.fail(f"mkdocs build --strict failed:\n{msg}")


# ===========================================================================
# Test 8: Sphinx build
# ===========================================================================

class TestSphinxBuild:
    """Sphinx must build HTML without errors."""

    @pytest.mark.docs_build
    def test_sphinx_html_build(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "sphinx",
                "-b", "html",
                "-W",  # treat warnings as errors
                "--keep-going",  # report all errors, not just first
                str(SOURCE_DIR),
                str(DOCS_DIR / "build" / "html"),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            stderr = result.stderr or ""
            stdout = result.stdout or ""
            output = stderr + stdout
            important = [
                l for l in output.splitlines()
                if any(kw in l.lower() for kw in ("error", "warning", "not found", "failed"))
            ]
            msg = "\n".join(important[:30]) if important else output[-2000:]
            pytest.fail(f"Sphinx build failed:\n{msg}")


# ===========================================================================
# Test 9: No broken image references
# ===========================================================================

class TestImageReferences:
    """Image references in markdown must point to existing files."""

    md_files = sorted(SOURCE_DIR.rglob("*.md")) if SOURCE_DIR.exists() else []

    @pytest.mark.parametrize(
        "md_file",
        md_files,
        ids=[str(f.relative_to(ROOT)) for f in md_files] if md_files else [],
    )
    def test_image_refs_exist(self, md_file: Path):
        text = md_file.read_text()
        broken = []
        # Standard markdown images: ![alt](path)
        for i, line in enumerate(text.splitlines(), 1):
            for m in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", line):
                src = m.group(2)
                if src.startswith(("http://", "https://")):
                    continue
                resolved = (md_file.parent / src).resolve()
                if not resolved.exists():
                    broken.append(f"  Line {i}: image '{src}' not found")
        # HTML img tags: <img src="...">
        for i, line in enumerate(text.splitlines(), 1):
            for m in re.finditer(r'<img[^>]+src=["\']([^"\']+)["\']', line):
                src = m.group(1)
                if src.startswith(("http://", "https://")):
                    continue
                resolved = (md_file.parent / src).resolve()
                if not resolved.exists():
                    broken.append(f"  Line {i}: image '{src}' not found")
        if broken:
            pytest.fail(
                f"Broken image refs in {md_file.relative_to(ROOT)}:\n"
                + "\n".join(broken)
            )


# ===========================================================================
# Test 10: mkdocs.yml is valid YAML loadable by mkdocs
# ===========================================================================

class TestMkDocsConfig:
    """mkdocs.yml must be parseable and contain required keys."""

    def test_mkdocs_yml_exists(self):
        assert MKDOCS_YML.exists(), "mkdocs.yml not found at project root"

    def test_mkdocs_yml_has_required_keys(self):
        text = MKDOCS_YML.read_text()
        for key in ("site_name", "nav", "theme", "markdown_extensions"):
            assert re.search(rf"^{key}:", text, re.MULTILINE), (
                f"mkdocs.yml missing required top-level key: {key}"
            )

    def test_sphinx_conf_exists(self):
        assert (SOURCE_DIR / "conf.py").exists(), "docs/source/conf.py not found"


# ===========================================================================
# Test 11: Sphinx conf.py is importable
# ===========================================================================

class TestSphinxConfig:
    """Sphinx conf.py must be importable without errors."""

    def test_conf_py_importable(self):
        conf_path = SOURCE_DIR / "conf.py"
        assert conf_path.exists()
        code = conf_path.read_text()
        try:
            compile(code, str(conf_path), "exec")
        except SyntaxError as e:
            pytest.fail(f"conf.py has syntax error: {e}")
