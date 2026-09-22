import cvxpy as cp
import numpy as np
from scipy.sparse import csc_array

from tomotok.inversions.solvers import Solver


class CvxpyNNLS(Solver):
    """
    Implements non-negative least squares solver using cvxpy.

    The cvxpy solver accepts sparse matrices.
    The solver can be configured to use different cvxpy solvers and verbosity levels.
    """
    def __init__(
            self,
            verbose: bool = False,
            cvxpy_solver = cp.CLARABEL,
        ):
        """
        Parameters
        ----------
        verbose : bool
            If True, enables verbose output from the cvxpy solver.
        cvxpy_solver : cvxpy solver
            The solver from cvxpy to be used. Default is cp.CLARABEL.
        """
        super().__init__()
        self.verbose = verbose
        self.cvxpy_solver = cvxpy_solver

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

        x = cp.Variable(a.shape[1])
        objective = cp.Minimize(cp.sum_squares(a @ x - b))
        constraints = [x >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.cvxpy_solver, verbose=self.verbose)
        return x.value
