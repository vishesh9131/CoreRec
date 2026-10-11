"""Intel Macs get NumPy 1.x-compatible pins; everyone else is unaffected (#212).

PyTorch's last x86_64 macOS wheel is 2.2.2, built against NumPy 1.x, so an
unpinned install there imported a NumPy 2 it can't run with.
"""

from pathlib import Path

import pytest

packaging = pytest.importorskip("packaging.requirements")
try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = pytest.importorskip("tomli")

DEPS = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["project"]["dependencies"]
PINS = {"numpy": "<2", "scipy": "<1.14", "pandas": "<2.3", "scikit-learn": "<1.6"}


def _active(env):
    out = {}
    for d in DEPS:
        r = packaging.Requirement(d)
        if r.marker is None or r.marker.evaluate(env):
            out.setdefault(r.name, []).append(str(r.specifier))
    return out


def test_intel_macs_get_numpy1_pins():
    active = _active({"sys_platform": "darwin", "platform_machine": "x86_64"})
    for name, pin in PINS.items():
        assert pin in active[name], name


@pytest.mark.parametrize("env", [{"sys_platform": "darwin", "platform_machine": "arm64"},
                                 {"sys_platform": "linux", "platform_machine": "x86_64"},
                                 {"sys_platform": "win32", "platform_machine": "AMD64"}])
def test_other_platforms_are_unpinned(env):
    active = _active(env)
    for name, pin in PINS.items():
        assert pin not in active[name], (name, env)
