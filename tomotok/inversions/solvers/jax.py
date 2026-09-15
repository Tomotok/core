# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
import numpy as np
import jax
import jax.numpy as jnp
from jax import config
from jax.numpy.linalg import solve
from jax.scipy.linalg import cho_factor, cho_solve
from jax.scipy.optimize import minimize
from scipy.sparse import spmatrix, sparray, issparse

from .base import Solver


class JaxCholesky(Solver):
    """Uses JAX's Cholesky decomposition for solving linear systems."""
    def __init__(self, enable_x64: bool = True) -> None:
        config.update("jax_enable_x64", enable_x64)
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
        factor = cho_factor(a)
        x = cho_solve(factor, b)
        return np.copy(x)


class JaxSolver(Solver):
    """Uses JAX's linear solver for solving linear systems."""
    def solve(
        self,
        a: np.ndarray | spmatrix | sparray,
        b: np.ndarray | spmatrix | sparray
    ) -> np.ndarray:
        if issparse(a):
            a = a.toarray()
        if issparse(b):
            b = b.toarray()
        x = solve(a, b)
        return np.copy(x)


class JaxNNLS(Solver):
    """
    Non-negative least squares solver using JAX.

    The solver supports different parametrizations for enforcing non-negativity, 
    including quadratic, exponential and softplus functions.
    """
    def __init__(self, parametrization: str = 'quadratic', init_values: float = 1.0) -> None:
        super().__init__()
        self.init_values = init_values
        if parametrization == 'quadratic':
            self._parametrization = lambda z: z**2
        elif parametrization == 'softplus':
            self._parametrization = jax.nn.softplus
        elif parametrization == 'exp':
            self._parametrization = jnp.exp
        else:
            raise ValueError(
                f"Unknown parametrization: {parametrization}. Supported: 'quadratic', 'softplus', 'exp'."
            )

    def solve(
        self,
        a: np.ndarray | sparray,
        b: np.ndarray | sparray 
    ):
        if issparse(a):
            a = a.toarray()
        if issparse(b):
            b = b.toarray()

        def loss_fn(z):
            x = self._parametrization(z)
            residual = jnp.dot(a, x) - b
            return 0.5 * jnp.sum(residual ** 2)

        @jax.jit
        def run_optimization(initial_z):
            res = minimize(loss_fn, initial_z, method='BFGS')
            return res.x

        z_init = jnp.ones(a.shape[1]) * self.init_values
        x = run_optimization(z_init)
        x = self._parametrization(x)
        return np.copy(x)
