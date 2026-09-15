import cvxpy as cp
import numpy as np
from scipy.sparse import csc_array

from ..base import Solver


class CvxpyNNLS(Solver):
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
        prob.solve(solver=cp.CLARABEL, verbose=False)
        return x.value
