# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from .bob import Bob
from .mfr import Mfr
from .fixed import Fixt


class CholeskyJax(object):
    """
    Template class overwriting MFR type inversions solving with jax.
    
    Requires jax to be installed in order to initialize properly.
    Uses jax.scipy.linalg cho_factor and cho_solve
    """
    def __init__(self, enable_x64: bool = True) -> None:
        from jax import config
        from jax.scipy.linalg import cho_factor, cho_solve
        config.update("jax_enable_x64", enable_x64)
        self._cho_factor = cho_factor
        self._cho_solve = cho_solve
        super().__init__()

    def invert(self, a: Union[np.ndarray, csr_matrix], b: np.ndarray) -> np.ndarray:
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
        factor = self._cho_factor(a)
        x = self._cho_solve(factor, b)
        return np.copy(x)


class CholeskyJaxMfr(CholeskyJax, Mfr):
    pass


class CholeskyJaxFixt(CholeskyJax, Fixt):
    pass


class Jax(object):
    """
    Template class overwriting inversion solving with jax.

    Requires jax to be installed in order to initialize properly.
    Uses jax.numpy.solve for inversion
    """
    def __init__(self) -> None:
        from jax.numpy.linalg import solve
        self._jax_solve = solve
        super().__init__()

    def invert(self, a: Union[np.ndarray, csr_matrix], b: np.ndarray) -> np.ndarray:
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
        x = self._jax_solve(a, b)
        return np.copy(x)


class JaxMfr(Jax, Mfr):
    pass

class JaxFixt(Jax, Fixt):
    pass


class CholeskyJaxBob(Bob):
    def __init__(self, enable_x64: bool = True) -> None:
        from jax import config
        from jax.scipy.linalg import cho_factor, cho_solve
        config.update("jax_enable_x64", enable_x64)
        self._cho_factor = cho_factor
        self._cho_solve = cho_solve
        super().__init__()

    def compute_coordinates(self, a: csr_matrix) -> csr_matrix:
        """
        Uses cholesky decomposition from JAX to solve the decomposition task

        Utilizes cho_factor and cho_solve from jax.scipy.linalg
        
        Parameters
        ----------
        a : scipy.sparse.csr_matrix
            square and positive definite matrix
        """
        factor = self._cho_factor(a.toarray())
        b = np.eye(*a.shape)
        x = self._cho_solve(factor, b)
        c = np.copy(x)
        c = csr_matrix(c)
        return c
