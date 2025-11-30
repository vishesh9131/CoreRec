# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CoreRec"
copyright = "2024, Vishesh Yadav"
author = "Vishesh Yadav"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Theme options for sphinx_book_theme
html_theme_options = {
    "repository_url": "https://github.com/vishesh9131/CoreRec",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs/source",
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
}

# Logo and favicon
# html_logo = "_static/logo.png"
# html_favicon = "_static/favicon.ico"

html_context = {
    "display_github": True,
    "github_user": "vishesh9131",
    "github_repo": "CoreRec",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

# Fix for Sphinx 8.x search index empty titles issue
def _patch_search_index():
    """Patch Sphinx search index to handle empty titles gracefully"""
    try:
        from sphinx.search import IndexBuilder
        
        original_freeze = IndexBuilder.freeze
        
        def patched_freeze(self):
            if not self._titles:
                # Return empty structure if no titles
                return {}, {}, {}, {}, {}
            return original_freeze(self)
        
        IndexBuilder.freeze = patched_freeze
    except (ImportError, AttributeError):
        pass  # If patching fails, let it proceed normally

# Apply the patch
_patch_search_index()
