from typing import Tuple
import numpy as np

from tomotok.tools.regularisation import regularisation_matrix

from .base import Solver


class Minimumfisher(object):
    def __init__(self, solver):
        self.solver = solver
        return
    
    def __call__(
        self, 
        data: np.ndarray, gmat, derivatives, errors,
        derivative_weights=None,
        mfi_num=3,
        compensate_negative_lines=False, 
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        """
        Inversion using the Minimum Fisher Inversion method.

        Parameters
        ----------
        data : np.ndarray
        gmat : scipy.sparse.spmatrix
        derivatives : list of scipy.sparse.spmatrix
            list of derivative matrices with shape (#nodes, #nodes)
        errors : np.ndarray
        derivative_weights : list of floats or list of array-like, optional
            anisotropy of derivatives, by default None
        mfi_num : int, optional
            number of minimum Fisher loops, by default 3
        compensate_negative_lines : bool, optional
            whether to balance the regularisation matrix, by default False
        kwargs
            Keyword arguments passed to the solver initialisation method

        Returns
        -------
        np.ndarray
            inversion result
        dict
            statistics of the inversion
        """
        g = np.ones(gmat.shape[1])
        tmp_stats = []

        for i in range(mfi_num):
            solver: Solver = self.solver(**kwargs)
            node_weights = 1 / g
            node_weights[g <= 0] = np.nanmax(node_weights)
            regularisation = regularisation_matrix(
                derivatives,
                derivative_weights,
                node_weights,
                compensate_negative_lines=compensate_negative_lines,
            )
            out, stats = solver(data, gmat, regularisation, errors)
            tmp_stats.append(stats)
            g = out

        statistics = {}
        for key in tmp_stats[0].keys():
            statistics[key] = [tmp_stats[i][key] for i in range(mfi_num)]
        return out, statistics

    def _mfi_loop(self, num, data, gmat, derivatives, errors, derivative_weights, **kwargs):
        solver = self.solver()
        node_weights = 1 / g
        node_weights[g <= 0] = np.nanmax(node_weights)
        regularisation = regularisation_matrix(derivatives, derivative_weights, node_weights)
        out, stats = solver(data, gmat, regularisation, errors, **kwargs)
        return out, stats


class FixedMinimumfisher(object):
    def __init__(self, solver):
        self.solver = solver
        return
    
    def __call__(
        self, data: np.ndarray, gmat, derivatives, parameters,
        derivative_weights=None, mfi_num=3, compensare_negative_lines=False, 
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        """
        Inversion using the Fixed Minimum Fisher Inversion method.

        Parameters
        ----------
        data : np.ndarray
        gmat : scipy.sparse.spmatrix
        derivatives : list of scipy.sparse.spmatrix
            list of derivative matrices with shape (#nodes, #nodes)
        errors : np.ndarray
        derivative_weights : list of floats or list of array-like, optional
            anisotropy of derivatives, by default None
        parameters : float or list of floats
            base 10 logarithms of regularisation parameters for each MFI loop
            if float, the same parameter is used for all MFI loops
        mfi_num : int, optional
            number of minimum Fisher loops, by default 3
        compensate_negative_lines : bool, optional
            whether to balance the regularisation matrix, by default False
        kwargs
            Keyword arguments passed to the solver initialisation method

        Returns
        -------
        np.ndarray
            inversion result
        dict
            statistics of the inversion
        """
        g = np.ones(gmat.shape[1])
        tmp_stats = []

        if isinstance(parameters, float):
            parameters = mfi_num * [parameters]
        elif len(parameters) != mfi_num:
            msg = 'Different number of regularisation parameters: {} and MFI loops: {}.'
            raise ValueError(msg.format(len(parameters), mfi_num))

        for i in range(mfi_num):
            solver: Solver = self.solver()
            node_weights = 1 / g
            node_weights[g <= 0] = np.nanmax(node_weights)
            regularisation = regularisation_matrix(derivatives, derivative_weights, node_weights, compensate_negative_lines=compensare_negative_lines)
            out, stats = solver(data, gmat, regularisation, **kwargs)
            tmp_stats.append(stats)
            g = out

        statistics = {}
        for key in tmp_stats[0].keys():
            statistics[key] = [tmp_stats[i][key] for i in range(mfi_num)]
        return out, statistics


class FloatQueue:
    def __init__(self, floats):
        self.floats = list(floats)

    def pop_first(self):
        if not self.floats:
            raise IndexError("No more floats in the queue.")
        return self.floats.pop(0)