"""Code in the published docs must import things that exist.

tests/test_docs.py already checks that documented Python *parses* and that
automodule/autoclass targets resolve. Neither catches the common case: a plain
``from corerec.x import Y`` in a fenced code block naming a module that was
renamed or deleted. When this was first measured, 74 of the 117 corerec modules
referenced across the docs were not importable -- most of them still using the
pre-rename ``contentFilterEngine`` / ``unionizedFilterEngine`` namespaces.

Scope is deliberately the pages mkdocs actually publishes. docs/source/ is a
second, Sphinx-shaped tree that the docs workflow builds but never deploys
(only ``mkdocs gh-deploy`` publishes), so its ~54 stale pages are a separate
cleanup and are not gated here.

KNOWN_STALE lists the pages that still reference removed APIs, so the backlog is
visible and cannot grow: a page not on the list must have working imports.
"""

import importlib
import re
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MKDOCS = ROOT / "mkdocs.yml"

IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(corerec[\w\.]*)\s+import|import\s+(corerec[\w\.]*))", re.M
)

# Placeholders that are meant to be filled in by the reader, not imported.
PLACEHOLDERS = {"corerec.engines.your_model"}

# Pages that reference removed APIs. Their code examples import modules that no
# longer exist (SVD, NMF, RLRMC, DeviceManager, the corerec.core.encoders
# classes, the corerec.utils logger/profiler/example_data helpers), so a reader
# copying them hits ModuleNotFoundError on the first line.
#
# Recorded rather than rewritten: an attempt to fix these in-place was reverted
# at the author's request. The listing keeps the problem visible and stops new
# breakage being added; fixing a page means deleting its entry here.
KNOWN_STALE = {
    "core/index.md",
    "engines/collaborative/index.md",
    "engines/index.md",
    "examples/index.md",
    "getting-started/architecture.md",
    "testing/index.md",
    "utilities/index.md",
}

_cache: dict = {}


def _importable(module: str) -> bool:
    if module not in _cache:
        try:
            importlib.import_module(module)
            _cache[module] = True
        except Exception:
            _cache[module] = False
    return _cache[module]


def _published_pages():
    if not MKDOCS.exists():
        return []
    nav = MKDOCS.read_text(errors="ignore")
    pages = []
    for path in sorted(DOCS.rglob("*.md")):
        if "build" in path.parts or "source" in path.parts:
            continue
        rel = path.relative_to(DOCS).as_posix()
        if rel in nav:
            pages.append(rel)
    return pages


PUBLISHED = _published_pages()


def _broken_imports(rel_path: str):
    text = (DOCS / rel_path).read_text(errors="ignore")
    modules = {m.group(1) or m.group(2) for m in IMPORT_RE.finditer(text)}
    return sorted(m for m in modules - PLACEHOLDERS if not _importable(m))


def test_docs_are_actually_published():
    """Guard against this whole file silently testing nothing."""
    assert len(PUBLISHED) > 20, f"only found {len(PUBLISHED)} published pages"


@pytest.mark.parametrize("rel_path", PUBLISHED, ids=PUBLISHED)
def test_published_page_imports_resolve(rel_path):
    broken = _broken_imports(rel_path)
    if rel_path in KNOWN_STALE:
        if not broken:
            pytest.fail(
                f"{rel_path} is listed in KNOWN_STALE but all its imports now "
                "resolve - remove it from the set."
            )
        pytest.xfail(f"{rel_path} still documents removed APIs: {broken}")
    assert not broken, (
        f"{rel_path} documents modules that cannot be imported: {broken}. "
        "Either fix the import path or drop the example."
    )


def test_known_stale_list_only_names_real_pages():
    """A typo'd entry would silently disable a check."""
    missing = [p for p in KNOWN_STALE if not (DOCS / p).is_file()]
    assert not missing, f"KNOWN_STALE names pages that do not exist: {missing}"
