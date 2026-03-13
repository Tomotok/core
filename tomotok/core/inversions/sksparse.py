# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
import numpy as np
from scipy import sparse

from .base import Engine


class CholmodEngine(Engine):
    def __init__(self):
        from sksparse.cholmod import cholesky
        self._cholesky = cholesky
        super().__init__()

    def solve(self, a: np.ndarray | sparse.sparray, b: np.ndarray | sparse.sparray) -> np.ndarray:
        if not sparse.issparse(a):
            a = sparse.csr_array(a)
        factor = self._cholesky(a)
        return factor(b)
