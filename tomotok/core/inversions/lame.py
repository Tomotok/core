# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Structure of classes is based on algorithms proposed by T. Odstrcil however without sparse optimization

T. Odstrcil et al., "Optimized tomography methods for plasma emissivity reconstruction at the
ASDEX Upgrade tokamak," Rev. Sci. Instrum., 87(12), 123505.
"""
import numpy as np
import scipy.sparse as sparse
from scipy.stats.mstats import mquantiles
from scipy.sparse.linalg import eigsh


class Algebraic(object):
    """
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

    def invert(self, sig, gmat, deriv, method=None, num=None):
        """
        Computes linear inversion using algebraic method.
        The inversion comprises of three stages:
            - decomposition (presolving) using only geometry and derivative matrices
            - searching for regularisation parameter
            - solving inversion using series expansion

        Parameters
        ----------
        sig : numpy.ndarray
        gmat : numpy.ndarray
        deriv : numpy.ndarray
        method : str, optional
            method used for regularization parameter computation, see method find_alpha
        num : int, optional
            use only num most significant vectors in series expansion

        Returns
        -------
        numpy.ndarray
            reconstructed emissivity vector with shape (#pix,)
        """
        self.u, self.s, self.v, = self.presolve(gmat, deriv)
        alpha = self.find_alpha(method)
        self.alpha = alpha
        g = self.series_expansion(alpha, sig, num=num)
        return g

    def __call__(self, data, gmat, dmats, errors, method=None, num=None):
        """
        Iterates all time slices in data and computes inversion using one of linear algebraic methods.

        Parameters
        ----------
        gmat : np.ndarray
            geometry matrix with shape (#chnls, #nodes)
        data : np.ndarray
            signal with shape (#slices, #chnls)
        dmats : 
            TODO
        errors : int, float or np.ndarray
            expected errors used to normalize data, if 1D shape should match #slices or #chnls, for 2D shape should match data shape
        h : np.ndarray
            regularisation matrix used in generalised decompositions
        method : str, optional
            specifies method used for regularization parameter computation, see method find_alpha
        num : int, optional
            sets number of largest vectors summed in the series expansion

        Returns
        -------
        numpy.ndarray
            results of inversion with shape (#timeslices, ny, nx)
        """
        nslices = data.shape[0]
        nchnls = data.shape[1]
        nnodes = gmat.shape[1]

        if np.ndim(errors) == 1:
            if errors.size == data.shape[0]:
                errors = errors.reshape(1, -1)
            elif errors.size == data.shape[1]:
                errors = errors.reshape(-1, 1)
            else:
                raise ValueError('Incompatible shape of errors {} with the data {}'.format(errors.shape, data.shape))
        
        errors = errors * np.ones_like(data)    
        data = data / errors

        res = np.empty((nslices, nnodes))

        danis = 1
        c1 = danis / (1 + danis)
        c2 = 1 / (1 + danis)
        # TODO support for list of dmat pairs
        n_der = len(dmats)
        hs = np.zeros((n_der, *dmats[0][0].shape))
        for i in range(n_der):
            # TODO
            tmp0 = dmats[i][0].T.dot(dmats[i][0])
            tmp1 = dmats[i][1].T.dot(dmats[i][1])
            hs[i, ...] = (c1 * tmp0 + c2 * tmp1).toarray()
        # h = np.sum(hs).toarray()
        h = hs.mean(axis=0)
        for i in range(data.shape[0]):
            signal = data[i, :].flatten()
            errs = np.diag(1/errors[i, :])
            gmat_nrm = errs.dot(gmat)
            # error_sp = sparse.diags(1/errors[i, :])
            # gmat_nrm = error_sp.dot(gmat).toarray()
            res[i] = self.invert(signal, gmat_nrm, h, method=method, num=num)
        return res

    def presolve(self, gmat, deriv):
        """
        Prepares matrices in standard form for series expansion

        Returns
        -------
        u, s, v : numpy.ndarray
        """
        raise NotImplementedError()

    def find_alpha(self, *args, **kwargs):
        """
        Finds regularisation parameter using estimate based on values of diagonal.

        Returns
        -------
        numpy.float64
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
        alpha : numpy.float64
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
        tmp = filters / s * self.u.T.dot(signal) * self.v
        g = tmp[:, :num].sum(axis=1)
        return g


class FastAlgebraic(Algebraic):
    def __init__(self):
        super().__init__()
        return

    def presolve(self, gmat, deriv):
        raise NotImplementedError()

    def find_alpha(self, method=None):
        """
        Finds regularisation parameter using linear estimate based on values of diagonal.

        Parameters
        ----------
        method : str {'mean', 'half', 'median', 'quantile', 'logmean}
            if None quantile method is used

        Returns
        -------
        numpy.float64
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
    def presolve(self, gmat, deriv):
        l_mat = np.linalg.cholesky(deriv)
        p = np.identity(l_mat.shape[0])
        l_inv = np.linalg.inv(l_mat)
        a = l_inv.dot(p).dot(gmat.T)
        u, s, vt = np.linalg.svd(a.T, full_matrices=False)
        v = vt.T
        v_tild = p.T.dot(l_inv.T).dot(v)
        return u, s, v_tild


class QrFastAlgebraic(FastAlgebraic):
    def presolve(self, gmat, deriv):
        l_mat = np.linalg.cholesky(deriv)
        p = np.identity(l_mat.shape[0])
        l_inv = np.linalg.inv(l_mat)
        a = l_inv.dot(p.dot(gmat.T))
        q1, d_roof, s = np.linalg.qr(a.dot(p))
        q2, r2 = np.linalg.qr(p.dot(s.T))
        m = d_roof.dot(r2.T).dot(np.linalg.inv(d_roof))
        r3, d3, q3 = np.linalg.qr(m)
        raise NotImplementedError('Not yet finished')


class GevFastAlgebraic(FastAlgebraic):
    def presolve(self, gmat, deriv):
        """
        Decomposes geometry and regularisation matrices to form suitable for series expansion.
        Uses generalised eigenvalue decomposition described in [1]_

        Parameters
        ----------
        gmat : numpy.ndarray
        deriv : numpy.ndarray
            regularisation matrix with shape (#nodes, #nodes)

        Returns
        -------
        u, s, v

        References
        ----------
        .. [1] L.C. Ingesson, "The Mathematics of Some Tomography Algorithms Used at JET," JET Joint Undertaking, 2000
        """
        c = gmat.T.dot(gmat)
        s, ev = eigsh(c.T, k=gmat.shape[0], M=deriv)
        # flip to have eigenvalues and vectors sorted from largest to smallest
        s = s[::-1]
        ev = ev[..., ::-1]

        s_sqrt = np.sqrt(s)

        u = gmat.dot(ev) / s_sqrt
        v = s_sqrt * ev
        return u, s, v
