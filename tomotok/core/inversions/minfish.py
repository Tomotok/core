import numpy as np

from tomotok.tools.regularisation import regularisation_matrix

from .base import Solver


class Minimumfisher(object):
    def __init__(self, solver):
        self.solver = solver
        return
    
    def __call__(self, data: np.ndarray, gmat, derivatives, errors, derivative_weights=None, mfi_num=3, **kwargs):
        g = np.ones(gmat.shape[1])
        tmp_stats = []

        for i in range(mfi_num):
            solver: Solver = self.solver()
            node_weights = 1 / g
            node_weights[g <= 0] = np.nanmax(node_weights)
            regularisation = regularisation_matrix(derivatives, derivative_weights, node_weights)
            out, stats = solver(data, gmat, regularisation, errors, **kwargs)
            tmp_stats.append(stats)
            g = out

        statistics = {}
        for key in tmp_stats[0].keys():
            statistics[key] = [tmp_stats[i][key] for i in range(mfi_num)]
        return out, statistics
