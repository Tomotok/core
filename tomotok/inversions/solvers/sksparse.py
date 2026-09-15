# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
import numpy as np
from scipy import sparse
from sksparse.cholmod import cholesky

from .base import Solver


class SksparseCholesky(Solver):
    def solve(self, a: np.ndarray | sparse.sparray, b: np.ndarray | sparse.sparray) -> np.ndarray:
        if not sparse.issparse(a):
            a = sparse.csr_array(a)
        factor = cholesky(a)
        x = factor(b)
        return x
