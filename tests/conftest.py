"""Shared pytest fixtures.

Several test modules load corerec source files directly with
spec_from_file_location and register the result in sys.modules under real
names ("corerec", "corerec.engines.content_based"). Those stand-ins are built
from a single file, so they have no __path__ and are not packages, and nothing
ever removed them. Once one of those tests ran, every later test that did a
genuine `import corerec.<anything>` failed with

    No module named 'corerec.multimodal'; 'corerec' is not a package

That is why test_online_serving.py passed 8/8 on its own and errored 8/8 in a
full run: pure ordering damage, not a bug in the code under test. CI never
caught it because it ran the suite as several cherry-picked invocations, so the
polluting tests and the victims were rarely in the same process.
"""

import sys

import pytest

import corerec  # noqa: F401  -- import the real package before anything stubs it

_REAL_MODULES = {
    name: mod
    for name, mod in sys.modules.items()
    if name == "corerec" or name.startswith("corerec.")
}


@pytest.fixture(autouse=True)
def restore_corerec_modules():
    """Put the real corerec modules back in sys.modules after every test.

    Only entries captured at import time are restored. Modules legitimately
    imported during a test are left alone: evicting those forces a re-import
    that rebuilds the classes, and then isinstance checks against the earlier
    objects fail (that alone broke 10 tests in test_all_production_models.py).
    """
    yield
    for name, real in _REAL_MODULES.items():
        if sys.modules.get(name) is not real:
            sys.modules[name] = real
