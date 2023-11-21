from typing import Union

import numpy as np
from jax import config
from jax.scipy.linalg import cho_factor, cho_solve
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix


config.update("jax_enable_x64", True)


class Jaxed(object):
    """
    Template class overwriting inversion solving with jax.
    Requires jax to be installed in order to initialize properly.

    Uses jax.scipy.linalg cho_factor and cho_solve
    """
    def __init__(self) -> None:
        super().__init__()
        self.cho_factor = cho_factor
        self.cho_solve = cho_solve

    def _make_cache(self, signals: ArrayLike, gmat: Union[ArrayLike, spmatrix]) -> None:
        super()._make_cache(signals, gmat)
        if isinstance(self._gdg, spmatrix):
            self._gdg = self._gdg.toarray()

    def invert(self, a: ArrayLike, b: ArrayLike) -> np.ndarray:
        r"""
        Finds solution of :math:`\mathbf{Ax}=\mathbf{b}` using jax.numpy.linalg.solve

        Parameters
        ----------
        a : array_like
            square and positive definite matrix
        b : array_like
            right hand side vector
        """
        factor = self.cho_factor(a)
        x = self.cho_solve(factor, b)
        return np.copy(x)
