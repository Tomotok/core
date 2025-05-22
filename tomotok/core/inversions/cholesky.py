from typing import List
from warnings import warn

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar

from .base import RegularisedSolver


class Cholesky(RegularisedSolver):
    """
    Solver of inverse problem with Phillips-Tikhonov regularisation using Cholesky decomposition.
    """

    def __init__(self):
        super().__init__()
        self._gmat = None
        self._signal = None
        self._gdg = None
        self._gdsig = None
        self._regularisation = None
        self._chi_sq = None

    def __call__(self, data, gmat, regularisation, errors):
        if len(gmat.shape) != 2:
            raise ValueError('Gmat must be a 2D array or matrix')
        if errors.shape != data.shape:
            raise ValueError('Data shape {} does not match errors shape {}.'.format(data.shape, errors.shape))

        if isinstance(errors, (int, float)):  # constant error estimate for all channels
            errors = np.full_like(data, errors)
        
        data_nrm = data / errors
        norms = sparse.diags(1/errors)
        gmat_nrm = norms @ gmat
        self._make_cache(data_nrm, gmat_nrm)
        self._regularisation = regularisation

        alpha, stats = self.determine_regularisation(
            regularisation,
            self._bounds,
            self._iter_max,
            self._tolerance,
        )
        g = self.invert(alpha)
        chi_sq = self._pearson_test(g)
        stats['chi_sq'] = chi_sq
        return g, stats

    def _make_cache(self, data, gmat) -> None:
        """
        Prepares cache for Cholesky solver.

        Parameters
        ----------
        data
            normalised data vector
        gmat
            normalised geometry matrix
        """
        self._signal = data
        self._gmat = gmat
        self._gdg = gmat.T @ gmat
        self._gdsig = gmat.T @ data

    @staticmethod
    def solve(a, b):
        r"""
        Finds solution of :math:`\mathbf{Ax}=\mathbf{b}` using scipy.sparse.linalg.spsolve
        """
        if isinstance(a, sparse.spmatrix):
            a = a.toarray()
        factor = cho_factor(a)
        return cho_solve(factor, b)

    def invert(self, alpha):
        """
        Inverts using provided regularisation parameter.

        Parameters
        ----------
        alpha : float
            regularisation parameter
        """
        mod_mat = self._gdg + alpha * self._regularisation
        g = self.solve(mod_mat, self._gdsig)
        return g


class PearsonCholesky(Cholesky):
    """
    Cholesky solver for the inversion problem using Pearson test to select regularisation parameter.
    """
    def __init__(
        self,
        bounds=(-30, 0),
        iter_max=10, 
        tolerance=1e-3,
    ):
        self._bounds = bounds
        self._iter_max = iter_max
        self._tolerance = tolerance
        super().__init__()

    def determine_regularisation(self):
        """
        Determines value of regularisation parameter using minimisation of Pearson test.
        
        The minimisation is done using the `minimize_scalar` function from `scipy.optimize`.

        Parameters
        ----------
        regularisation : sparse.spmatrix
            The regularisation matrix.
        bounds : tuple
            The bounds for the regularisation parameter.
        iter_max : int
            The maximum number of iterations.
        tolerance : float
            The tolerance for convergence.

        Returns
        -------
        sparse.spmatrix
            The regularisation matrix.
        dict
            A dictionary containing statistics about the inversion process.
        """
        # TODO: write custom optimisation routine to avoid recalculating optimal solution to get chi sq
        res = minimize_scalar(
            self._test_regularization,
            method='bounded',
            bounds=self._bounds,
            options={'maxiter': self._iter_max, 'xatol': self._tolerance},
        )
        if res.status == 1:
            warn('Maximum number of iteration in regularisation parameter search. Consider increasing iter_max.')
        stats = dict(
            iter_num=res.nfev,
            logalpha=res.x,
        )
        alpha = 10**res.x
        return alpha, stats

    def _pearson_test(self, g):
        r"""
        Computes retrofit and residuum :math:`\chi^2` using pearson test

        .. math ::
            \chi^2 = \frac{1}{M} \sum_{i}^{M} \left(\tilde{\mathbf{f}} - \tilde{\mathbf{T}} \cdot \mathbf{g} \right)_i^2

        Parameters
        ----------
        g : numpy.ndarray
            vector of tested emissivity

        Returns
        -------
        float
        """
        retrofit = self._gmat @ g
        misfit = retrofit - self._signal
        misfit_sq = np.power(misfit, 2)
        chisq = np.average(misfit_sq)
        return chisq

    def _test_regularization(self, logalpha):
        """
        Function passed to minimisation function used for finding regularisation parameter value.

        Inverses signals using given regularisation parameter and computes chi2 test

        Stores pearson test result in attribute last_chi.

        Parameters
        ----------
        logalpha : float
            natural logarithm of regularisation parameter

        Returns
        -------
        abs(chi2 - 1) : float
            1D Euclidean distance from ideal Pearson test result
        """
        alpha = 10**logalpha
        g = self.invert(alpha)
        chi2 = self._pearson_test(g)
        return abs(chi2 - 1)

class FixedCholesky(Cholesky):
    """
    Cholesky solver for the inversion problem using fixed regularisation parameter.
    """
    def __init__(self, parameters: List[float]):
        """
        Parameters
        ----------
        parameters : list of float
            List of 10 based logarithm of regularisation parameters.
        """
        super().__init__()
        self._parameters = parameters

    def determine_regularisation(self):
        alpha = 10**self._parameters.pop(0)
        stats = dict(
            iter_num=0,
            logalpha=self._parameters[0],
        )
        return alpha, stats
