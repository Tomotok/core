# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
import numpy as np
from scipy.sparse import spmatrix, sparray, issparse

from .base import Engine


class CholeskyJaxEngine(Engine):
    def __init__(self, enable_x64: bool = True) -> None:
        from jax import config
        from jax.scipy.linalg import cho_factor, cho_solve
        config.update("jax_enable_x64", enable_x64)
        self._cho_factor = cho_factor
        self._cho_solve = cho_solve
        super().__init__()

    def solve(
        self,
        a: np.ndarray | spmatrix | sparray,
        b: np.ndarray | spmatrix | sparray
    ) -> np.ndarray:
        if issparse(a):
            a = a.toarray()
        if issparse(b):
            b = b.toarray()
        factor = self._cho_factor(a)
        x = self._cho_solve(factor, b)
        return np.copy(x)


class JaxEngine(Engine):
    def __init__(self) -> None:
        from jax.numpy.linalg import solve
        self._jax_solve = solve
        super().__init__()

    def solve(
        self,
        a: np.ndarray | spmatrix | sparray,
        b: np.ndarray | spmatrix | sparray
    ) -> np.ndarray:
        if issparse(a):
            a = a.toarray()
        if issparse(b):
            b = b.toarray()
        x = self._jax_solve(a, b)
        return np.copy(x)
