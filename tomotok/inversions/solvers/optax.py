import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial

from scipy.sparse import issparse, sparray

from .base import Solver


class OptaxNNLS(Solver):
    def __init__(self, iterations: int = 5_000):
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
    Dense matrices only, requires reasonable estimate on learning rate and parameters.
    """
    def __init__(
            self,
            max_iter_num: int = 10_000,
            learning_rate: float = 0.1,
            tolerance: float = 1e-8,
        ):
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
