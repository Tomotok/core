# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Structure of classes is based on algorithms proposed by T. Odstrcil however without sparse optimization

T. Odstrcil et al., "Optimized tomography methods for plasma emissivity reconstruction at the
ASDEX Upgrade tokamak," Rev. Sci. Instrum., 87(12), 123505.
"""
from typing import Union, List, Tuple
from warnings import warn

import numpy as np
import scipy.sparse as sparse
from scipy.stats.mstats import mquantiles
from scipy.sparse.linalg import eigsh


Derivative_type = Union[sparse.spmatrix, List[sparse.spmatrix]]


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
            self, data: np.ndarray, gmat: sparse.spmatrix, regularisation: sparse.spmatrix, 
            method: str = None, num: int = None
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
        gmat : numpy.ndarray
        regularisation : numpy.ndarray
            regularisation matrix
        method : str, optional
            method used for regularization parameter computation, see method find_alpha
        num : int, optional
            use only num most significant vectors in series expansion

        Returns
        -------
        numpy.ndarray
            reconstructed emissivity vector with shape (#pix,)
        """
        self.u, self.s, self.v, = self.decompose(gmat, regularisation)
        alpha = self.find_alpha(method)
        self.alpha = alpha
        g = self.series_expansion(alpha, data, num=num)
        return g

    def __call__(
            self, data: np.ndarray, gmat: sparse.spmatrix, derivatives: Derivative_type, errors: np.ndarray, 
            derivative_weights: Union[int, float, List[float]] = None, method: str = None, num: int = None
            ) -> Tuple[np.ndarray, List[dict]]:
        """
        Iterates all time slices in data and computes inversion using one of linear algebraic methods.

        Parameters
        ----------
        data : numpy.ndarray
            signal with shape (#time slices, #chnls)
        gmat : numpy.ndarray
            geometry matrix with shape (#chnls, #nodes)
        derivatives : list
            list of tuples with derivative matrix pairs
        errors : int, float or numpy.ndarray
            expected errors used to normalize data, with shape (#slices,), (#chnls,) or (#time slices, #chnls) 
        aniso : float, optional
            anisotropic factor making the first 
        method : str, optional
            specifies method used for regularization parameter computation, see method find_alpha
        num : int, optional
            sets number of largest vectors summed in the series expansion

        Returns
        -------
        numpy.ndarray
            results of inversion with shape (#timeslices, ny, nx)
        list of dicts
            inversion statistics for each time slice
        """
        nslices = data.shape[0]
        nchnls = data.shape[1]
        if nchnls != gmat.shape[0]:
            raise ValueError('Different number of channels in data and gmat')
        nnodes = gmat.shape[1]

        data_ndim = np.ndim(data)
        if data_ndim == 0:
            raise ValueError('Data must be at least an array of values.')
        elif data_ndim == 1:  # flat data, assume one time slice
            data = data.reshape(1, -1)
        elif data_ndim > 2:
            raise ValueError('Data array has too many dimension.')

        err_ndim = np.ndim(errors)
        if err_ndim == 0:  # constant errors
            errors = np.full_like(data, errors)
        elif err_ndim == 1:  # flat errors, check for matching shape
            if errors.size == nslices:
                errors = errors.reshape(-1, 1)
            elif errors.size == nchnls:
                errors = errors.reshape(1, -1)
            else:
                raise ValueError('Incompatible shape of errors {} with the data {}'.format(errors.shape, data.shape))
            errors = errors * np.ones_like(data)  # broadcast to right shape
        elif err_ndim > 2:
            raise ValueError('Errors array has too many dimensions.')

        if errors.shape != data.shape:
            raise ValueError('Data shape {} does not match errors shape {}.'.format(data.shape, errors.shape))

        data = data / errors

        res = np.empty((nslices, nnodes))

        reg = self.regularisation_matrix(derivatives, derivative_weights)
        stats = []
        for i in range(data.shape[0]):
            signal = data[i, :].flatten()
            norms = sparse.diags(1/errors[i, :])
            gmat_nrm = norms @ gmat
            res[i] = self.invert(signal, gmat_nrm, reg, method=method, num=num)
            stats.append({'alpha': self.alpha})
        return res, stats

    def regularisation_matrix(
            self, derivatives: Derivative_type, derivative_weights=None, node_weights=None
        ) -> sparse.csc_matrix:
        """
        Computes regularisation matrix from derivatives matrices.

        Parameters
        ----------
        derivatives : _type_
            _description_
        aniso : int, optional
            _description_, by default 1

        Returns
        -------
        _type_
            _description_
        """
        if isinstance(derivatives, sparse.spmatrix):
            derivatives = [derivatives]
        if derivative_weights is None:
            derivative_weights = [1] * len(derivatives)
        elif isinstance(derivative_weights, (int, float)):
            derivative_weights = [derivative_weights] * len(derivatives)
        try:
            assert len(derivative_weights) == len(derivatives)
        except AssertionError:
            raise ValueError('Derivative weights must have same length as derivatives')
        except TypeError:
            raise TypeError('Derivative weights must be a number or a list of numbers')
        if node_weights is None:
            node_weights = [1]
        node_weights = sparse.diags(node_weights, shape=derivatives[0].shape)
        total = sum(derivative_weights)
        regularisation = sparse.csr_matrix(derivatives[0].shape)
        for dw, dmat in zip(derivative_weights, derivatives):
            regularisation += dw / total * dmat.T @ node_weights @ dmat
        return regularisation

    def presolve(self, gmat, deriv):
        """
        .. deprecated:: 1.1
            Use :meth:`decompose`  
        """
        self.decompose(gmat, deriv)
        warn('Presolve method deprecated by decompose method since 1.1', DeprecationWarning)

    def decompose(self, gmat, regularisation):
        """
        Prepares matrices in standard form for series expansion

        Returns
        -------
        u, s, v : numpy.ndarray
        """
        raise NotImplementedError()

    def find_alpha(self, *args, **kwargs):
        """
        Finds regularisation parameter.

        Returns
        -------
        float
            regularisation parameter value
        """
        raise NotImplementedError()

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
        tmp = filters / s * self.u.T @ signal * self.v
        g = tmp[:, :num].sum(axis=1)
        return g


class FastAlgebraic(Algebraic):
    """
    A base class for linear algebraic methods using fast regularisation parameter estimation.
    
    The regularisation parameter estimate is based on the diagonal matrix obtained by decomposition.
    """
    def __init__(self):
        super().__init__()
        return

    def decompose(self, gmat, regularisation):
        raise NotImplementedError()

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


class SvdFastAlgebraic(FastAlgebraic):
    def decompose(
            self, gmat: sparse.csr_matrix, regularisation: sparse.csc_matrix
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gmat = gmat.toarray()
        regularisation = regularisation.toarray()
        l_mat = np.linalg.cholesky(regularisation)
        p = np.identity(l_mat.shape[0])
        l_inv = np.linalg.inv(l_mat)
        a = (l_inv @ p) @ gmat.T
        u, s, vt = np.linalg.svd(a.T, full_matrices=False)
        v = vt.T
        v_tild = (p.T @ l_inv.T) @ v
        return u, s, v_tild


class QrFastAlgebraic(FastAlgebraic):
    def __init__(self):
        raise NotImplementedError('Not yet finished')

    def decompose(self, gmat, regularisation):
        l_mat = np.linalg.cholesky(regularisation)
        p = np.identity(l_mat.shape[0])
        l_inv = np.linalg.inv(l_mat)
        a = l_inv.dot(p.dot(gmat.T))
        q1, d_roof, s = np.linalg.qr(a.dot(p))
        q2, r2 = np.linalg.qr(p.dot(s.T))
        m = d_roof.dot(r2.T).dot(np.linalg.inv(d_roof))
        r3, d3, q3 = np.linalg.qr(m)
        raise NotImplementedError('Not yet finished')


class GevFastAlgebraic(FastAlgebraic):
    def decompose(self, gmat: sparse.csr_matrix, regularisation: sparse.csc_matrix):
        """
        Decomposes geometry and regularisation matrices to form suitable for series expansion.
        
        Uses generalised eigenvalue decomposition scheme described by L. C. Ingesson in [GEV]_

        Parameters
        ----------
        gmat : numpy.ndarray
        regularisation : numpy.ndarray
            regularisation matrix with shape (#nodes, #nodes)

        Returns
        -------
        u, s, v

        References
        ----------
        .. [GEV] L.C. Ingesson, "The Mathematics of Some Tomography Algorithms Used at JET," JET Joint Undertaking, 2000
        """
        gdg = gmat.T @ gmat
        s, ev = eigsh(gdg, k=gmat.shape[0], M=regularisation)

        # flip to have eigenvalues and vectors sorted from largest to smallest
        s = s[::-1]
        ev = ev[..., ::-1]

        s_sqrt = np.sqrt(s)

        u = gmat.dot(ev) / s_sqrt
        v = s_sqrt * ev
        return u, s, v
