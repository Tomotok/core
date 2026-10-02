Installation
============

Tomotok requires Python 3.10 or newer.

Tomotok is available from the Python Package Index (PyPI). It can be installed using pip::

    pip install tomotok

To install from source, clone the `GitHub repository <https://github.com/Tomotok/core>`_ and run the following command in its root::

    pip install .

The default branch is development.
It contains the latest version of the code; backward compatibility is not guaranteed and testing is limited.
If you want to use the latest stable version, switch to the corresponding tag or to the ``stable`` branch, which corresponds to the latest version available on PyPI.

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

The solvers using optional dependencies are not imported by :mod:`tomotok.inversions.solvers` and have to be imported from their submodules:

.. list-table::
    :header-rows: 1

    * - Extra
      - Solvers
    * - ``jax``
      - :class:`~tomotok.inversions.solvers.jax.JaxCholesky`, :class:`~tomotok.inversions.solvers.jax.JaxSolver`, :class:`~tomotok.inversions.solvers.jax.JaxNNLS`
    * - ``optax``
      - :class:`~tomotok.inversions.solvers.optax.OptaxNNLS`, :class:`~tomotok.inversions.solvers.optax.OptaxNN`
    * - ``sksparse``
      - :class:`~tomotok.inversions.solvers.sksparse.SksparseCholesky`
    * - ``cvxpy``
      - :class:`~tomotok.inversions.solvers.cvxpy.CvxpyNNLS`

The extras and solvers listed here correspond to the ``development`` branch and may not be available in older releases.

Development installation
-------------------------

To contribute to Tomotok or work with an editable checkout, clone the `github repository
<https://github.com/Tomotok/core>`_ and run the following commands from the repository root::

    python -m pip install --upgrade pip
    python -m pip install -e ".[docs]"

Optional solver backends can be combined with the editable install, e.g. ``pip install -e ".[jax]"``.

After installation, run the test suite to verify that everything works::

    python -m unittest discover -s tests -t . -p "test_*.py"

See ``CONTRIBUTING.md`` in the repository for the full contribution guidelines, including docstring
conventions and the documentation build process.

Upgrading from version 1
------------------------

The subpackages were moved from ``tomotok.core`` directly to the ``tomotok`` namespace, e.g. ``tomotok.core.inversions`` is now :mod:`tomotok.inversions`. The ``tomotok.core`` package is deprecated and kept only for compatibility; it re-exports :mod:`tomotok.geometry`, :mod:`tomotok.inversions` and :mod:`tomotok.regularisations` and emits a :class:`DeprecationWarning` on import. New code should import these packages directly.
