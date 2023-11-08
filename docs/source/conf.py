# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os


project = "ARLIN - Assured Reinforcement Learning Model Interrogation"
copyright = "2023, MITRE"
author = "Alex Tapley"

version = "1.0"

# The full version, including alpha/beta/rc tags
if os.environ.get("CI_COMMIT_TAG"):
    release = os.environ["CI_COMMIT_TAG"]
else:
    release = "latest"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "stable-baselines3": ("https://stable-baselines3.readthedocs.io/en/master/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# Autodoc settings
autodoc_typehints = "description"

# Autoapi settings
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"
autoapi_type = "python"
autoapi_dirs = ["../../arlin/"]

templates_path = ["_templates"]
exclude_patterns = ["**/_tests"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_logo = "../images/arlin_logo.png"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
