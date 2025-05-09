# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Structure of classes is based on algorithms proposed by T. Odstrcil however without sparse optimization

T. Odstrcil et al., "Optimized tomography methods for plasma emissivity reconstruction at the
ASDEX Upgrade tokamak," Rev. Sci. Instrum., 87(12), 123505.
"""
from typing import List, Tuple
from warnings import warn

import numpy as np
import scipy.sparse as sparse
from scipy.optimize import minimize_scalar
from scipy.stats.mstats import mquantiles
from scipy.sparse.linalg import eigsh


class Algebraic(object):
    """
    A base class for algebraic inversion methods using linear regularisation.

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
        self.u = None
        self.s = None
        self.v = None
        self.alpha = None
        return

    def invert(
            self, data: np.ndarray, gmat: sparse.spmatrix, regularisation: sparse.spmatrix, num: int = None, 
            *args, **kwargs
        ) -> np.ndarray:
        """
        Computes linear inversion using algebraic method.
        The inversion comprises of three stages:

            - decomposition (presolving) using only geometry and derivative matrices
            - searching for regularisation parameter
            - solving inversion using series expansion

        Parameters
        ----------
        data : numpy.ndarray
        gmat : sparse.spmatrix
            geometry matrix with shape (#channels, #nodes)
        regularisation : sparse.spmatrix
            regularisation matrix
        num : int, optional
            use only `num` most significant vectors in series expansion
        *args
            additional positional arguments passed to method determining regularisation parameter
        **kwargs
            additional keyword arguments passed to method determining regularisation parameter

        Returns
        -------
        numpy.ndarray
            reconstructed emissivity vector with shape (#pix,)

        See Also
        --------
        find_alpha : method for finding regularisation parameter
        series_expansion : method for computing emissivity from decomposed matrices
        """
        self.decompose(gmat, regularisation)
        # TODO: consider method regularize instead of find_alpha
        alpha = self.find_alpha(*args, **kwargs)
        self.alpha = alpha
        g = self.series_expansion(alpha, data, num=num)
        return g

    def __call__(
            self, data: np.ndarray, gmat: sparse.spmatrix, regularisation: sparse.spmatrix, errors: np.ndarray, num: int = None,
            *args, **kwargs
            ) -> Tuple[np.ndarray, List[dict]]:
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

        signal = data.flatten()
        norms = sparse.diags(1/errors)

        gmat_nrm = norms @ gmat
        res = self.invert(signal, gmat_nrm, regularisation, num=num, *args, **kwargs)
        stats = {'alpha': self.alpha}
        return res, stats

    def decompose(self, gmat, regularisation, *args, **kwargs):
        """
        Prepares matrices used in the series expansion.

        This method should be implemented in derived class.
        The matrices are stored in the class attributes `u`, `s`, and `v`.

        Returns
        -------
        u, s, v : numpy.ndarray
        """
        raise NotImplementedError('Decomposition should be implemented in derived class.')

    def find_alpha(self, *args, **kwargs):
        """
        Finds regularisation parameter.

        Returns
        -------
        float
            regularisation parameter value
        """
        raise NotImplementedError('Regularisation parameter estimation should be implemented in derived class.')

    def series_expansion(self, alpha, signal, num=None):
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
        signal : numpy.ndarray
            vector with channel signal
        num : int
            number of columns used for series expansion

        Returns
        -------
        numpy.ndarray
            results of inversion
        """
        s = self.s.reshape(1, -1)  # create row vector from diagonal matrix
        s_sq = np.square(s)
        filters = 1 / (1 + alpha / s_sq)
        tmp = filters / s * (self.u.T @ signal) * self.v
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


class FastAlgebraic(object):
    """
    A class for fast regularisation parameter estimation in linear algebraic methods.
    
    The regularisation parameter estimate is based on the diagonal matrix obtained by decomposition.
    This class should be combined with an Algebraic subclass implementing the decompose method.
    """
    def __init__(self):
        super().__init__()
        return

    def find_alpha(self, method: str='quantile'):
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
        return alpha**2


class FastSvdAlgebraic(FastAlgebraic, SvdAlgebraic):
    """
    A class for linear algebraic methods using SVD decomposition and fast regularisation parameter estimation.
    
    The regularisation parameter estimate is based on the diagonal matrix obtained by decomposition.
    """
    def __init__(self):
        super().__init__()
        return


class FastGevAlgebraic(FastAlgebraic, GevAlgebraic):
    """
    A class for linear algebraic methods using GEV decomposition and fast regularisation parameter estimation.
    
    The regularisation parameter estimate is based on the diagonal matrix obtained by decomposition.
    """
    def __init__(self):
        super().__init__()
        return


class PearsonAlgebraic(object):
    """
    A class for linear algebraic methods using Pearson decomposition and fast regularisation parameter estimation.
    
    The regularisation parameter estimate is based on the diagonal matrix obtained by decomposition.
    """
    def __init__(self):
        raise NotImplementedError('Not yet finished')
        super().__init__()
        return

    def _pearson(self, data_nrm):
        return

    def find_alpha(self):
        minimize_scalar()