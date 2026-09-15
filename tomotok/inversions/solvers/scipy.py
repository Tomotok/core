import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import nnls, lsq_linear

from .base import Solver


class CholeskySolver(Solver):
    """
    Scipy based engine using Cholesky decomposition to solve linear systems.

    Implementation based on dense matrices, sparse ones are converted to dense.
    """
    def __init__(self, check_finite: bool = False):
        super().__init__()
        self._check_finite = check_finite

    def solve(
        self,
        a: np.ndarray | sparse.sparray,
        b: np.ndarray | sparse.sparray
    ) -> np.ndarray | sparse.sparray:
        """Sparse matrices are converted to dense arrays before decomposition."""
        if sparse.issparse(a):
            a = a.toarray()
        if sparse.issparse(b):
            b = b.toarray()
        factor = cho_factor(a, check_finite=self._check_finite)
        return cho_solve(factor, b)


class NNLSSolver(Solver):
    """Scipy based engine using non-negative least squares to solve linear systems."""
    def __init__(self):
        super().__init__()

    def solve(
        self,
        a: np.ndarray | sparse.sparray,
        b: np.ndarray | sparse.sparray
    ) -> np.ndarray | sparse.sparray:
        """Sparse matrices are converted to dense arrays before decomposition."""
        if sparse.issparse(a):
            a = a.toarray()
        if sparse.issparse(b):
            b = b.toarray()
        x, _ = nnls(a, b)
        return x


class LSQSolver(Solver):
    """
    Scipy based engine using bounded least squares to solve linear systems.

    This engine works with sparse matrices, converting dense matrices to sparse format.

    It uses lsa_linear from scipy.optimize to solve the problem with bounds.
    The default bounds are set to (0, inf) to ensure non-negativity of the solution.
    
    """
    def __init__(self, bounds: tuple[float, float] | None = (0, np.inf)):
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
