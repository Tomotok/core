# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html for the available options.
import os
import sys

from packaging.version import Version

# make the package importable from the repository without installation
sys.path.insert(0, os.path.abspath('../../'))

with open(os.path.join(os.path.dirname(__file__), '../../tomotok/VERSION')) as version_file:
    ver = version_file.read().strip()

# -- Project information -----------------------------------------------------

project = 'tomotok'
copyright = '2021–2026, Institute of Plasma Physics of the Czech Academy of Sciences'
author = 'J. Svoboda, J. Cavalier, O. Ficker, M. Imríšek and J. Mlynář'
version = '{0.major}.{0.minor}'.format(Version(ver))  # short X.Y version
release = ver  # full version including development tags
language = 'en'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'numpydoc',
]
templates_path = ['_templates']

# private paths of types used in annotations that have no public documentation target
nitpick_ignore = [
    ('py:class', 'numpy._typing.ArrayLike'),
    ('py:class', 'h5py._hl.group.Group'),
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Autodoc and autosummary -------------------------------------------------

autodoc_default_options = {
    'members': True,
    'special-members': '__call__',
}
autodoc_member_order = 'bysource'
# constructor parameters are documented in __init__ docstrings (see CONTRIBUTING.md)
autoclass_content = 'both'
# optional solver backends (see pyproject.toml extras) are mocked, so their modules
# can be documented even when these packages are not installed in the docs build environment
autodoc_mock_imports = ['jax', 'optax', 'sksparse', 'cvxpy']
# API reference pages in docs/source/api list the documented objects, stub pages are generated
autosummary_generate = True

# -- Numpydoc ----------------------------------------------------------------

numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
# methods are already listed by autodoc members option
numpydoc_show_class_members = False

# -- HTML output -------------------------------------------------------------

html_theme = 'sphinx_book_theme'
html_title = f'Tomotok {release}'
pygments_style = 'sphinx'
