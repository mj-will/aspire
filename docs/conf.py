"""Sphinx configuration for the aspire documentation."""

from __future__ import annotations

import os
import sys

# Ensure the package can be imported when building the docs locally
ROOT = os.path.abspath("..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import aspire  # noqa: E402

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "aspire"
copyright = "2025, Michael J. Williams"
author = "Michael J. Williams"
version = aspire.__version__
release = aspire.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "autoapi.extension",
]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Configure autoapi -------------------------------------------------------

autoapi_type = "python"
autoapi_dirs = ["../src/aspire/"]
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "imported-members",
    "show-inheritance",
    "show-module-summary",
    "undoc-members",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = "aspire"
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/mj-will/aspire",
    "repository_branch": "main",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
