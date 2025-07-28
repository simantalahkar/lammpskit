# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# Import the package to get version information
try:
    import lammpskit

    release = lammpskit.__version__
except ImportError:
    release = "1.2.1"

project = "LAMMPSKit"
copyright = "2025, Simanta Lahkar"
author = "Simanta Lahkar"
version = release

# Project description
html_title = "LAMMPSKit Documentation"
html_short_title = "LAMMPSKit"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # Re-enabled for hybrid approach
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
]

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Mock imports for packages that may not be available during documentation builds
autodoc_mock_imports = [
    "ovito",
    "ovito.io",
    "ovito.modifiers",
]

# Continue on import errors
autodoc_inherit_docstrings = True
autodoc_preserve_defaults = True

# Napoleon settings for Google/NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autosummary settings
autosummary_generate = True
autosummary_generate_overwrite = True

# Type hints configuration
typehints_fully_qualified = False
always_document_param_types = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

templates_path = []
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Additional HTML context
html_context = {
    "display_github": True,
    "github_user": "simantalahkar",
    "github_repo": "lammpskit",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

# Custom sidebar
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html", 
        "searchbox.html",
        "donate.html",
    ]
}
