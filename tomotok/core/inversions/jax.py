# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix

from .mfr import Mfr
from .fixed import Fixt


class Jaxed(object):
    """
    Template class overwriting inversion solving with jax.
    Requires jax to be installed in order to initialize properly.

    Uses jax.scipy.linalg cho_factor and cho_solve
    """
    def __init__(self) -> None:
        from jax import config
        from jax.scipy.linalg import cho_factor, cho_solve
        config.update("jax_enable_x64", True)
        self.cho_factor = cho_factor
        self.cho_solve = cho_solve
        super().__init__()

    def invert(self, a: Union[ArrayLike, spmatrix], b: ArrayLike) -> np.ndarray:
        r"""
        Finds solution of :math:`\mathbf{Ax}=\mathbf{b}` using jax.scipy

        Parameters
        ----------
        a : array_like
            square and positive definite matrix
        b : array_like
            right hand side vector
        """
        if isinstance(a, spmatrix):
            a = a.toarray()
        factor = self.cho_factor(a)
        x = self.cho_solve(factor, b)
        return np.copy(x)


class JaxedMfr(Jaxed, Mfr):
    pass


class JaxedFixt(Jaxed, Fixt):
    pass
