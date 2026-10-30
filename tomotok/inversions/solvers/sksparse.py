# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
import numpy as np
from scipy import sparse
from sksparse.cholmod import cho_factor

from .base import Solver


class SksparseCholesky(Solver):
    """
    Solves linear systems using the sparse Cholesky decomposition from scikit-sparse (CHOLMOD).

    The system matrix has to be symmetric positive definite. Dense inputs are converted to sparse format.
    Requires the optional dependency ``scikit-sparse``.
    """
    def solve(self, a: np.ndarray | sparse.sparray, b: np.ndarray | sparse.sparray) -> np.ndarray:
        if not sparse.issparse(a):
            a = sparse.csr_array(a)
        factor = cho_factor(a)
        x = factor.solve(b)
        return x
