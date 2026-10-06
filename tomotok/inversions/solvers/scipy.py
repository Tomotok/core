# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences.
# Licensed under the EUPL-1.2 or later.
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import nnls, lsq_linear

from .base import Solver


class CholeskySolver(Solver):
    """
    Solves linear systems using the Cholesky decomposition from SciPy.

    The system matrix has to be symmetric positive definite.
    Implementation based on dense matrices, sparse inputs are converted to dense arrays.
    """
    def __init__(self, check_finite: bool = False):
        """
        Parameters
        ----------
        check_finite : bool, optional
            whether to check that the input matrices contain only finite numbers, by default False
        """
        super().__init__()
        self._check_finite = check_finite

    def solve(
        self,
        a: np.ndarray | sparse.sparray,
        b: np.ndarray | sparse.sparray
    ) -> np.ndarray | sparse.sparray:
        if sparse.issparse(a):
            a = a.toarray()
        if sparse.issparse(b):
            b = b.toarray()
        factor = cho_factor(a, check_finite=self._check_finite)
        return cho_solve(factor, b)


class NNLSSolver(Solver):
    """
    Solves linear systems using the non-negative least squares method from SciPy.

    Finds the non-negative solution minimising the residual of the linear system.
    When used by :class:`~tomotok.inversions.tikhonov.Tikhonov`, it is applied to the regularised normal equations,
    so the result is the non-negative least squares solution of that system, which in general differs from
    the minimum of the Tikhonov functional constrained to non-negative values.
    Implementation based on dense matrices, sparse inputs are converted to dense arrays.
    """
    def __init__(self):
        super().__init__()

    def solve(
        self,
        a: np.ndarray | sparse.sparray,
        b: np.ndarray | sparse.sparray
    ) -> np.ndarray | sparse.sparray:
        if sparse.issparse(a):
            a = a.toarray()
        if sparse.issparse(b):
            b = b.toarray()
        x, _ = nnls(a, b)
        return x


class LSQSolver(Solver):
    """
    Solves linear systems using the bounded least squares method from SciPy.

    Uses :func:`scipy.optimize.lsq_linear` to find the solution within given bounds.
    Works with sparse matrices, dense inputs are converted to sparse format.
    """
    def __init__(self, bounds: tuple[float, float] | None = (0, np.inf)):
        """
        Parameters
        ----------
        bounds : tuple of two floats, optional
            lower and upper bound of the solution, by default (0, inf), which ensures non-negativity of the solution
        """
        super().__init__()
        self._bounds = bounds

    def solve(
        self,
        a: np.ndarray | sparse.sparray,
        b: np.ndarray | sparse.sparray
    ) -> np.ndarray | sparse.sparray:
        if not sparse.issparse(a):
            a = sparse.csr_array(a)
        if sparse.issparse(b):
            b = b.toarray()
        if b.ndim > 1:
            b = b.ravel()
        res = lsq_linear(a, b, bounds=self._bounds)
        return res.x
