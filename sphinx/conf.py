# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../hetero2d'))
sys.path.insert(0, os.path.abspath('../hetero2d/workflow'))
sys.path.insert(0, os.path.abspath('../hetero2d/io'))
sys.path.insert(0, os.path.abspath('../hetero2d/manipulate'))
sys.path.insert(0, os.path.abspath('../hetero2d/utility'))
sys.path.insert(0, os.path.abspath('../hetero2d/fireworks'))
sys.path.insert(0, os.path.abspath('../hetero2d/firetasks'))
# -- Project information -----------------------------------------------------

project = 'Hetero2d'
copyright = '2021, Tara Maria Boland'
author = 'Tara Maria Boland'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
autodoc_member_order = 'bysource'
autodoc_mock_imports = ["fireworks", "firetasks"]
autoclass_content = 'both'
autodoc_default_flags = ['members', 'undoc-members', 'private-members', 'special-members', 'inherited-members', 'show-inheritance']
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
