# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Structure of classes is based on algorithms proposed by T. Odstrcil however without sparse optimization

T. Odstrcil et al., "Optimized tomography methods for plasma emissivity reconstruction at the
ASDEX Upgrade tokamak," Rev. Sci. Instrum., 87(12), 123505.
"""
from typing import Tuple
from warnings import warn

import numpy as np
import scipy.sparse as sparse
from scipy.optimize import minimize_scalar
from scipy.stats.mstats import mquantiles
from scipy.sparse.linalg import eigsh

from .base import Solver


class Algebraic(Solver):
    """
    A base class for solvers base on algebraic inversion methods.

    Attributes
    ----------
    u : numpy.ndarray
        decomposition matrix with shape (#channels, #channels)
    s : numpy.ndarray
        diagonal from a decomposition matrix S with shape (#channels, )
    v : numpy.ndarray
        decomposition matrix with shape (#nodes, #channels)    
    """
    def __init__(self):
        self.u: np.ndarray = None
        self.s: np.ndarray = None
        self.v: np.ndarray = None
        return

    def __call__(
            self, data: np.ndarray, gmat: sparse.spmatrix, regularisation: sparse.spmatrix, errors: np.ndarray, num: int = None,
            *args, **kwargs
            ) -> Tuple[np.ndarray, dict]:
        """
        Computes linear inversion using algebraic method.
        The inversion comprises of three stages:

            - decomposition (presolving) using only geometry and derivative matrices
            - searching for regularisation parameter
            - solving inversion using series expansion

        Parameters
        ----------
        data : numpy.ndarray
            signal with shape (,#chnls)
        gmat : numpy.ndarray
            geometry matrix with shape (#chnls, #nodes)
        regularisation : sparse.spmatrix
            regularisation matrix
        errors : int, float or numpy.ndarray
            expected errors used to normalize data, with shape (, #chnls) 
        num : int, optional
            sets number of largest vectors summed in the series expansion
        args
            additional positional arguments passed to method determining regularisation parameter
        kwargs
            additional keyword arguments passed to method determining regularisation parameter

        Returns
        -------
        numpy.ndarray
            results of inversion with
        dict
            inversion statistics for each time slice
        
        See Also
        --------
        find_alpha : method for finding regularisation parameter
        series_expansion : method for computing emissivity from decomposed matrices
        """
        if isinstance(errors, (int, float)):  # constant error estimate for all channels
            errors = np.full_like(data, errors)

        if errors.shape != data.shape:
            raise ValueError('Data shape {} does not match errors shape {}.'.format(data.shape, errors.shape))

        data = data / errors
        norms = sparse.diags(1/errors)

        gmat_nrm = norms @ gmat
        if self.s is None:
            self.decompose(gmat_nrm, regularisation)
        alpha, stats = self.find_alpha(*args, **kwargs)
        res = self.series_expansion(alpha, data, num=num)
        return res, stats

    def decompose(self, gmat, regularisation, *args, **kwargs):
        """
        Prepares matrices used in the series expansion.

        This method should be implemented in derived class.
        The matrices are stored in the class attributes `u`, `s`, and `v`.
        """
        warn('Decomposition is not implemented in base class. '
             'This method should be implemented in derived class.', UserWarning)
        return

    def find_alpha(self, *args, **kwargs) -> Tuple[float, dict]:
        """
        Finds regularisation parameter.

        Returns
        -------
        float
            regularisation parameter value
        dict
            statistics of the regularisation parameter estimation
        """
        raise NotImplementedError('Regularisation parameter estimation should be implemented in derived class.')

    def series_expansion(self, alpha, data=None, num=None):
        r"""
        Computes emissivity :math:`g` from decomposed vectors using

        .. math::
            \mathbf{g}(\alpha) = \sum_{i=1}^{m} \frac{k_{i} (\alpha)}{S_{ii}}
            \left( \mathbf{U}^T \cdot \mathbf{f} \cdot \tilde{\mathbf{V}} \right) {}_{*i},

        where :math:`k_i(\alpha)` are so called filtering factors computed using following formula

        .. math::
            k_{i}(\alpha) = \left(1 + \frac{\alpha}{S_{ii}^2} \right)^{-1}
        

        Parameters
        ----------
        alpha : float
            regularisation parameter
        data : numpy.ndarray, optional
            vector with data to be inverted, if not provided, the data stored in the class are used
        num : int
            number of columns used for series expansion

        Returns
        -------
        numpy.ndarray
            results of inversion
        """
        if data is None:
            data = self._data
        s = self.s.reshape(1, -1)  # create row vector from the diagonal of the diagonal matrix
        s_sq = np.square(s)
        filters = 1 / (1 + alpha / s_sq)
        tmp = filters / s * (self.u.T @ data) * self.v
        g = tmp[:, :num].sum(axis=1)
        return g


class SvdAlgebraic(Algebraic):
    def decompose(
            self, gmat: sparse.csr_matrix, regularisation: sparse.csc_matrix
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gmat = gmat.toarray()
        regularisation = regularisation.toarray()
        l_mat = np.linalg.cholesky(regularisation)
        p = np.identity(l_mat.shape[0])
        l_inv = np.linalg.inv(l_mat)
        a = (l_inv @ p) @ gmat.T
        self.u, self.s, vt = np.linalg.svd(a.T, full_matrices=False)
        v = vt.T
        v_tild = (p.T @ l_inv.T) @ v
        self.v = v_tild
        return


class GevAlgebraic(Algebraic):
    def decompose(self, gmat: sparse.csr_matrix, regularisation: sparse.csc_matrix, eigenvalues: int = None):
        """
        Decomposes geometry and regularisation matrices to form suitable for series expansion.
        
        Uses generalised eigenvalue decomposition scheme described by L. C. Ingesson in [GEV]_

        Parameters
        ----------
        gmat : numpy.ndarray
            geometry matrix with shape (#channels, #nodes), should not be normalised for this method
        regularisation : numpy.ndarray
            regularisation matrix with shape (#nodes, #nodes)
        eigenvalues : int, optional
            number of eigenvalues to be computed, by default None, which means number of nodes eigenvalues will be computed

        Returns
        -------
        u, s, v

        References
        ----------
        .. [GEV] L.C. Ingesson, "The Mathematics of Some Tomography Algorithms Used at JET," JET Joint Undertaking, 2000
        """
        eigenvalues = eigenvalues or gmat.shape[0]
        gdg = gmat.T @ gmat
        s, ev = eigsh(gdg, k=eigenvalues, M=regularisation)

        # flip to have eigenvalues and vectors sorted from largest to smallest
        self.s = s[::-1]
        ev = ev[..., ::-1]

        s_sqrt = np.sqrt(s)

        self.u = (gmat @ ev) / s_sqrt
        self.v = s_sqrt * ev
        return


# class QrAlgebraic(Algebraic):
#     def __init__(self):
#         raise NotImplementedError('Not yet finished')

#     def decompose(self, gmat, regularisation):
#         l_mat = np.linalg.cholesky(regularisation)
#         p = np.identity(l_mat.shape[0])
#         l_inv = np.linalg.inv(l_mat)
#         a = l_inv.dot(p.dot(gmat.T))
#         q1, d_roof, s = np.linalg.qr(a.dot(p))
#         q2, r2 = np.linalg.qr(p.dot(s.T))
#         m = d_roof.dot(r2.T).dot(np.linalg.inv(d_roof))
#         r3, d3, q3 = np.linalg.qr(m)


class FastAlgebraic(Algebraic):
    """
    A class for fast regularisation parameter estimation in linear algebraic methods.
    
    The regularisation parameter estimate is based on the diagonal matrix obtained by decomposition.
    This class should be combined with an Algebraic subclass implementing the decompose method.
    """
    def __init__(self):
        super().__init__()
        return

    def find_alpha(self, method: str='quantile') -> Tuple[float, dict]:
        """
        Finds regularisation parameter using linear estimate based on values of diagonal.

        Parameters
        ----------
        method : str {'mean', 'half', 'median', 'quantile', 'logmean}
            selects method for finding regularisation parameter value, default is quantile

        Returns
        -------
        float
            squared value found by estimation method
        dict
            statistics of the regularisation parameter estimation
        """
        if method is None:
            method = 'quantile'
        if method == 'mean':
            alpha = self.s.mean()
        elif method == 'half':
            alpha = self.s.max() / 2
        elif method == 'median':
            alpha = mquantiles(self.s, prob=0.5, alphap=0, betap=1)[0]
        elif method == 'quantile':
            quant = 2 / np.e
            alpha = mquantiles(self.s, prob=quant, alphap=0, betap=1)[0]
        elif method == 'logmean':
            log_s = np.log10(self.s)
            alpha = np.power(10, log_s.mean())
        else:
            raise ValueError('Unrecognized option for regularisation parameter estimation: {}'.format(method))
        stats = {}
        stats['method'] = method
        stats['alpha'] = alpha
        stats['logalpha'] = np.log10(alpha)
        stats['alpha_sq'] = alpha**2
        return alpha**2, stats


class FastSvdAlgebraic(FastAlgebraic, SvdAlgebraic):
    """
    Uses SVD decomposition and fast regularisation parameter estimation.
    
    The regularisation parameter estimate is based on the diagonal matrix obtained by decomposition.
    """
    pass


class FastGevAlgebraic(FastAlgebraic, GevAlgebraic):
    """
    Uses GEV decomposition and fast regularisation parameter estimation.
    
    The regularisation parameter estimate is based on the diagonal matrix obtained by decomposition.
    """
    pass


class PearsonAlgebraic(Algebraic):
    """
    A class for linear algebraic methods using Pearson decomposition and fast regularisation parameter estimation.
    
    The regularisation parameter estimate is based on the diagonal matrix obtained by decomposition.
    """
    def __init__(self):
        super().__init__()
        return

    def __call__(
        self, 
        data: np.ndarray, 
        gmat: sparse.spmatrix, 
        regularisation: sparse.spmatrix, 
        errors: np.ndarray, 
        num: int = None,
        bounds: Tuple[float, float] = (-20, 0),
        iter_max: int = 13,
        tolerance: float = 1e-3,
    ) -> Tuple[np.ndarray, dict]:
        self._data = data
        self._gmat =  gmat
        self._regularisation = regularisation
        self._errors = errors
        self._alpha = None
        self._chisq = None
        out, stats = super().__call__(
            data, gmat, regularisation, errors, num=num,
            bounds=bounds, iter_max=iter_max, tolerance=tolerance
        )
        return out, stats

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
        misfit = retrofit - self._data
        misfit_sq = np.power(misfit, 2)
        self._chisq = np.average(misfit_sq)
        return self._chisq

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
        alpha = 10 ** logalpha
        g = self.series_expansion(alpha, self._data)
        chi2 = self._pearson_test(g)
        return abs(chi2 - 1)

    def find_alpha(self, bounds, iter_max: int = 13, tolerance: float = 1e-3) -> Tuple[float, dict]:
        """
        Finds regularisation parameter using pearson test.

        Uses bounded method of scipy.optimize.minimize_scalar to find the optimal regularisation parameter.
        The regularisation parameter is not optimized directly, but its 10 based logarithm is used instead.
        The function to be minimized is the absolute difference between the chi-squared statistic and 1.

        Parameters
        ----------
        bounds : tuple
            lower and upper bounds for the regularisation parameter
        iter_max : int
            maximum number of iterations for the minimization algorithm
        tolerance : float
            tolerance for the minimization algorithm
        
        Returns
        -------
        float
            regularisation parameter value
        dict
            statistics of the regularisation parameter estimation

        See Also
        --------
        _test_regularization : method for testing regularisation parameter
        """
        res = minimize_scalar(
            self._test_regularization,
            method='bounded',
            bounds=bounds,
            options={'maxiter': iter_max, 'xatol': tolerance},
        )
        if res.status == 1:
            warn('Maximum number of iteration in regularisation parameter search. Consider increasing iter_max.')
        stats = dict(
            iter_num=res.nfev,
            logalpha=res.x,
            chi2=self._chisq,
        )
        return 10**res.x, stats

    def invert(self, alpha):
        """
        Inverts the signal using the given regularisation parameter.

        Parameters
        ----------
        alpha : float
            regularisation parameter

        Returns
        -------
        numpy.ndarray
            inverted signal
        """
        return self.series_expansion(alpha)

class PearsonGevAlgebraic(PearsonAlgebraic, GevAlgebraic):
    """
    Uses GEV decomposition to solve the inverse problem and Pearson test to estimate the regularisation parameter.
    """
    pass


class PearsonSvdAlgebraic(PearsonAlgebraic, SvdAlgebraic):
    """
    Uses SVD decomposition to solve the inverse problem and Pearson test to estimate the regularisation parameter.
    """
    pass
