import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial

from scipy.sparse import issparse, sparray

from .base import Solver


class OptaxNNLS(Solver):
    """
    Solves linear systems using the non-negative least squares method from Optax.

    Uses ``optax.nnls`` compiled by JAX. Sparse inputs are converted to dense arrays.
    Requires the optional dependencies ``jax`` and ``optax``.
    """
    def __init__(self, iterations: int = 5_000):
        """
        Parameters
        ----------
        iterations : int, optional
            number of iterations of the NNLS algorithm, by default 5000
        """
        super().__init__()
        self.iterations = iterations

    def solve(
            self,
            a: np.ndarray | sparray, 
            b: np.ndarray | sparray,
    ):
        if issparse(a):
            a = a.toarray()
        if issparse(b):
            b = b.toarray()
        self.a = jnp.array(a)
        self.b = jnp.array(b)
        nnls_compiled = jax.jit(partial(optax.nnls, iters=self.iterations))
        x = nnls_compiled(self.a, self.b)
        return np.copy(x)


class OptaxNN(Solver):
    """
    Solves linear systems with non-negativity constraint by gradient-based optimisation using Optax.

    Minimises the squared residual using the Adam optimiser with projection to non-negative values.
    Works with dense matrices only, sparse inputs are converted to dense arrays.
    Requires a reasonable estimate of the learning rate and other parameters.
    Requires the optional dependencies ``jax`` and ``optax``.
    """
    def __init__(
            self,
            max_iter_num: int = 10_000,
            learning_rate: float = 0.1,
            tolerance: float = 1e-8,
        ):
        """
        Parameters
        ----------
        max_iter_num : int, optional
            maximum number of optimisation steps, by default 10000
        learning_rate : float, optional
            learning rate of the Adam optimiser, by default 0.1
        tolerance : float, optional
            optimisation stops when the change of the loss between two steps is smaller, by default 1e-8
        """
        super().__init__()
        self.max_iter_num = max_iter_num
        self.tolerance = tolerance
        self.optimizer = optax.chain(
            optax.adam(learning_rate=learning_rate),
            optax.keep_params_nonnegative()
        )

    def _loss_fn(self, x):
        residual = jnp.dot(self.a, x) - self.b
        return jnp.sum(residual ** 2)

    @jax.jit
    def step(self, x, opt_state):
        loss, grads = jax.value_and_grad(self._loss_fn)(x)
        updates, opt_state = self.optimizer.update(grads, opt_state, x)
        x = optax.apply_updates(x, updates)
        return x, opt_state, loss

    def solve(
            self,
            a: np.ndarray | sparray, 
            b: np.ndarray | sparray,
    ):
        if issparse(a):
            a = a.toarray()
        if issparse(b):
            b = b.toarray()
        self.a = jnp.array(a)
        self.b = jnp.array(b)
        
        x_init = jnp.zeros(a.shape[1])
        opt_state = self.optimizer.init(x_init)

        
        x = x_init
        prev_loss = float('inf')
        self.losses = np.zeros(self.max_iter_num)

        for i in range(self.max_iter_num):
            x, opt_state, loss_val = self.step(x, opt_state)
            self.losses[i] = float(loss_val)

            # Check for convergence
            loss_change = abs(prev_loss - float(loss_val))
            if loss_change < self.tolerance:
                break
            prev_loss = float(loss_val)
        return np.copy(x)
