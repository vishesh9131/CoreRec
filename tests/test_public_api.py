"""The public surface should contain what CoreRec means to expose, and nothing else.

Three things this pins down, each of which was wrong:

  - `corerec.vish_graphs` injected twenty misspelled aliases of one function
    into its globals(), so a single helper was nearly half the module's public
    names and none of them could ever be removed without breaking a caller.
  - Several modules had no __all__, so everything they imported -- torch,
    numpy, pandas, matplotlib, csv, multiprocessing -- was reachable as public
    API and would land in a caller's namespace on a star-import.
  - `corerec.data` was listed in corerec.__all__ with a 0-byte __init__.py, so
    it advertised itself and exported nothing.
"""

import types

import pytest

import corerec


REMOVED_MISSPELLINGS = [
    "scale_and_save_matrix",
    "scal_save_matrices",
    "scaleandsavematrices",
    "scaling_save_matrix",
    "scale_n_save_matricies",
]


# Submodules that legitimately require an optional extra. They stay in __all__
# because they are part of the package; they just need `pip install corerec[x]`.
OPTIONAL_SUBMODULES = {"serving": "serving", "multimodal": "transformers"}


def test_every_advertised_name_resolves():
    """Nothing in corerec.__all__ may be missing or None."""
    broken = []
    for name in corerec.__all__:
        try:
            if getattr(corerec, name) is None:
                broken.append(f"{name} (None)")
        except AttributeError as exc:
            if name in OPTIONAL_SUBMODULES:
                continue  # covered by test_optional_submodule_names_its_extra
            broken.append(f"{name} ({exc})")
    assert not broken, f"corerec.__all__ advertises unusable names: {broken}"


@pytest.mark.parametrize("name,extra", sorted(OPTIONAL_SUBMODULES.items()))
def test_optional_submodule_names_its_extra(name, extra):
    """A missing extra must say which one, not just 'no attribute'.

    In a base install `corerec.serving` raised
    "module 'corerec' has no attribute 'serving'", which reads like the feature
    does not exist rather than like fastapi/pydantic are not installed.
    """
    try:
        getattr(corerec, name)
    except AttributeError as exc:
        assert f"corerec[{extra}]" in str(exc), (
            f"corerec.{name} failed without naming its extra: {exc}"
        )


@pytest.mark.parametrize("alias", REMOVED_MISSPELLINGS)
def test_misspelled_aliases_are_gone(alias):
    """They must raise, and the error must name the real function."""
    import corerec.vish_graphs as vg

    with pytest.raises(AttributeError, match="scale_and_save_matrices"):
        getattr(vg, alias)


def test_the_real_function_still_exists():
    import corerec.vish_graphs as vg

    assert callable(vg.scale_and_save_matrices)
    assert "scale_and_save_matrices" in vg.__all__


@pytest.mark.parametrize(
    "module_name,forbidden",
    [
        ("corerec.vish_graphs", ["csv", "np", "nx", "plt", "sp", "time", "multiprocessing"]),
        ("corerec.visualization", ["np", "pd", "plt", "torch"]),
        ("corerec.metrics", ["np", "nx"]),
    ],
)
def test_star_import_does_not_leak_dependencies(module_name, forbidden):
    """`from corerec.x import *` must not dump third-party modules on the caller."""
    namespace = {}
    exec(f"from {module_name} import *", namespace)  # noqa: S102 - that is the thing under test
    leaked = sorted(n for n in forbidden if n in namespace)
    assert not leaked, f"{module_name} star-import leaked {leaked}"


@pytest.mark.parametrize("module_name", ["corerec.vish_graphs", "corerec.visualization", "corerec.metrics"])
def test_modules_declare_all(module_name):
    import importlib

    module = importlib.import_module(module_name)
    assert hasattr(module, "__all__"), f"{module_name} has no __all__"
    assert module.__all__, f"{module_name}.__all__ is empty"
    for name in module.__all__:
        assert hasattr(module, name), f"{module_name}.__all__ names missing {name}"


def test_corerec_data_exports_its_datasets():
    """It was advertised in corerec.__all__ with an empty __init__."""
    from corerec import data

    assert data.__all__, "corerec.data exports nothing"
    assert "RecommendationDataset" in data.__all__
    from corerec.data import RecommendationDataset

    assert isinstance(RecommendationDataset, type)


def test_no_bare_module_objects_in_top_level_all():
    """Submodules are fine in __all__; bare third-party modules are not."""
    third_party = {"np", "numpy", "pd", "pandas", "torch", "plt", "matplotlib", "nx", "networkx"}
    leaked = [
        n for n in corerec.__all__
        if n in third_party and isinstance(getattr(corerec, n, None), types.ModuleType)
    ]
    assert not leaked, f"corerec.__all__ re-exports third-party modules: {leaked}"
