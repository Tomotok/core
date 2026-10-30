tomotok.inversions
==================

.. automodule:: tomotok.inversions
    :no-members:

.. currentmodule:: tomotok.inversions

Inversion methods
-----------------

.. autosummary::
    :toctree: generated/

    Tikhonov
    MinimumFisherRegularisation
    SvdAlgebraic
    GevAlgebraic
    Bob

Regularisation parameter selectors
----------------------------------

.. autosummary::
    :toctree: generated/

    PearsonSelector
    FixedSelector
    FastSelector

Solvers
-------

.. module:: tomotok.inversions.solvers

.. currentmodule:: tomotok.inversions.solvers

Solvers of linear systems used by :class:`~tomotok.inversions.Tikhonov` and :class:`~tomotok.inversions.Bob`. :class:`Solver` is the base class of all solvers, the other solvers below are based on SciPy and are named after the numerical method.

.. autosummary::
    :toctree: generated/

    Solver
    CholeskySolver
    NNLSSolver
    scipy.LSQSolver

.. currentmodule:: tomotok.inversions

.. autosummary::
    :toctree: generated/

    SparseInvSolver

.. currentmodule:: tomotok.inversions.solvers

Solvers with optional dependencies are named after their backend (JAX, Optax, scikit-sparse, CVXPY) and the numerical method. They have to be imported from their submodules, see :doc:`../install`.

.. autosummary::
    :toctree: generated/

    jax.JaxCholesky
    jax.JaxSolver
    jax.JaxNNLS
    optax.OptaxNNLS
    optax.OptaxNN
    sksparse.SksparseCholesky
    cvxpy.CvxpyNNLS

Base classes
------------

.. currentmodule:: tomotok.inversions

Base classes for implementation of new inversion methods and regularisation parameter selectors.

.. autosummary::
    :toctree: generated/

    base.Inversion
    base.RegularisedInversion
    base.RegularisationSelector
    lame.Algebraic
