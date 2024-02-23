# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix, csc_matrix

from .bob import Bob
from .mfr import Mfr
from .fixed import Fixt


class CholeskyJax(object):
    """
    Template class overwriting inversion solving with jax.
    Requires jax to be installed in order to initialize properly.

    Uses jax.scipy.linalg cho_factor and cho_solve
    """
    def __init__(self, enable_x64: bool = True) -> None:
        from jax import config
        from jax.scipy.linalg import cho_factor, cho_solve
        config.update("jax_enable_x64", enable_x64)
        self.cho_factor = cho_factor
        self.cho_solve = cho_solve
        super().__init__()

    def invert(self, a: Union[ArrayLike, csr_matrix], b: ArrayLike) -> np.ndarray:
        r"""
        Finds solution of :math:`\mathbf{Ax}=\mathbf{b}` using jax.scipy

        Parameters
        ----------
        a : array_like
            square and positive definite matrix
        b : array_like
            right hand side vector
        """
        if isinstance(a, (csr_matrix, csc_matrix)):
            a = a.toarray()
        factor = self.cho_factor(a)
        x = self.cho_solve(factor, b)
        return np.copy(x)


class JaxedMfr(CholeskyJax, Mfr):
    pass


class JaxedFixt(CholeskyJax, Fixt):
    pass


class Jax(object):
    """
    Template class overwriting
    inversion solving with jax.
    Requires jax to be installed in order to initialize properly.

    Uses jax.numpy.solve for inversion
    """
    def __init__(self) -> None:
        from jax.numpy.linalg import solve
        self.jax_solve = solve

    def invert(self, a: Union[ArrayLike, csr_matrix], b: ArrayLike) -> np.ndarray:
        r"""
        Finds solution of :math:`\mathbf{Ax}=\mathbf{b}` using jax.numpy.linalg.solve

        Parameters
        ----------
        a : array_like
            square and positive definite matrix
        b : array_like
            right hand side vector
        """
        if isinstance(a, (csr_matrix, csc_matrix)):
            a = a.toarray()
        x = self.jax_solve(a, b)
        return np.copy(x)


class JaxMfr(Jax, Mfr):
    pass


class JaxedBob(Bob):
    def compute_coefficients(self, a: csr_matrix, enable_x64=True) -> csr_matrix:
        """
        Uses JAX to solve the decomposition
        
        Parameters
        ----------
        a : scipy.sparse.csr_matrix
            square and positive definite matrix
        enable_x64 : bool, optional
            toggles 64bit precision, by default True
        """
        from jax import config
        from jax.scipy.linalg import cho_factor, cho_solve
        config.update("jax_enable_x64", enable_x64)
        factor = cho_factor(a.toarray())
        b = np.eye(*a.shape)
        x = cho_solve(factor, b)
        c = np.copy(x)
        c = csr_matrix(c)
        return c
