Installation
============

Tomotok requires Python 3.10 or newer.

Tomotok is available from python package index. It can be installed using pip by running command::

    pip install tomotok

If pip is not available it can be installed from source using files downloaded from a `github repository <https://github.com/Tomotok>`_ by running command::

    pip install .

in the folder with the source code.

The default branch is development.
It contains the latest version of the code, the backward compatibility is not guaranteed and with limited testing.
If you want to use the latest stable version, you can switch to the corresponding tag or `stable` branch that is the latest version available on pip.

Documentation built from the ``development`` branch, reflecting the latest unreleased changes, is published at
`tomotok.github.io/core <https://tomotok.github.io/core/>`_.

Optional solver backends
-------------------------

The core inversion algorithms only need ``numpy``, ``scipy``, ``matplotlib`` and ``h5py``, which are installed
automatically. Some solver classes rely on additional, optional dependencies that can be pulled in as extras::

    pip install "tomotok[jax]"          # jax
    pip install "tomotok[optax]"        # jax + optax
    pip install "tomotok[sksparse]"     # scikit-sparse (Cholmod)
    pip install "tomotok[cvxpy]"        # cvxpy
    pip install "tomotok[all-solvers]"  # all of the above

Development installation
-------------------------

To contribute to Tomotok or work with an editable checkout, clone the `github repository
<https://github.com/Tomotok/core>`_ and run the following commands from the repository root::

    python -m pip install --upgrade pip
    python -m pip install -e .
    python -m pip install -r requirements.txt

Optional solver backends can be combined with the editable install, e.g. ``pip install -e ".[jax]"``.

After installation, run the test suite to verify that everything works::

    python -m unittest discover -s tests -t . -p "test_*.py"

See ``CONTRIBUTING.md`` in the repository for the full contribution guidelines, including docstring
conventions and the documentation build process.
