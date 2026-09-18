import cvxpy as cp
import numpy as np
from scipy.sparse import csc_array, dia_array

from tomotok.inversions.solvers import Solver


class CvxpyNNLS(Solver):
    """
    Implements non-negative least squares solver using cvxpy.

    The cvxpy solver accepts sparse matrices.
    Normalization is applied to avoid differences in orders of magnitude between rows, which can improve convergence.
    The solver can be configured to use different cvxpy solvers and verbosity levels.
    """
    def __init__(
            self,
            verbose: bool = False,
            cvxpy_solver = cp.CLARABEL,
            normalize: bool = True,
        ):
        """
        Parameters
        ----------
        verbose : bool
            If True, enables verbose output from the cvxpy solver.
        cvxpy_solver : cvxpy solver
            The solver from cvxpy to be used. Default is cp.CLARABEL.
        normalize : bool
            If True, normalizes the input matrix to avoid differences in orders of magnitude between rows.
        """
        self.verbose = verbose
        self.cvxpy_solver = cvxpy_solver
        self.normalize = normalize

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        if not isinstance(value, bool):
            raise TypeError("verbose must be a bool.")
        self._verbose = value

    def solve(
        self,
        a: np.ndarray | csc_array,
        b: np.ndarray
    ):
        if not isinstance(a, csc_array):
            a = csc_array(a)

        # normalize to avoid differences in orders of magnitude between rows
        if self.normalize:
            norms = 1/a.max(1).toarray()
            dn = dia_array((norms, 0), shape=a.shape)
            a = dn @ a
            b = b * norms

        x = cp.Variable(a.shape[1])
        objective = cp.Minimize(cp.sum_squares(a @ x - b))
        constraints = [x >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.cvxpy_solver, verbose=self.verbose)
        return x.value
