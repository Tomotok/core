from numpy import ndarray
from numpy.typing import ArrayLike
from scipy.sparse import sparray


class Solver:
    """Base class for solving a system of linear equations."""
    def __init__(self, sparse: bool = False):
        self.sparse = sparse

    def solve(self, a: ArrayLike | sparray, b: ArrayLike | sparray) -> ndarray:
        r"""
        Solves the linear system :math:`\mathbf{Ax}=\mathbf{b}`.

        Parameters
        ----------
        a : array_like or sparse array
            System of equations matrix to be solved
        b : array_like
            right hand side vector or matrix (in case of multiple time slices)

        Returns
        -------
        numpy.ndarray
            The solution of the linear system.
        """
        raise NotImplementedError("This method should be defined by subclasses.")

    def __call__(self, a: ArrayLike | sparray, b: ArrayLike | sparray) -> ndarray:
        return self.solve(a, b)
