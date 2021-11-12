# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Minimun Fisher Regularisation implemented based on M. Anton and J. Mlynar

M. Anton et al., "X-ray tomography on the TCV tokamak.", Plasma Phys. Control. Fusion  38.11 (1996): 1849.

J. Mlynar et al., "Current research into applications of tomography for fusion diagnostics." J. Fusion Energy 38.3 (2019): 458-466
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize_scalar


class Mfr(object):
    r"""
    Inverses provided data using Minimum Fisher Regularization scheme.

    Attributes
    ----------
    last_chi : float
        person test result of the last computed inversion
    logalpha : float
        :math:`\log(\alpha)` logarithm of regularisation parameter
    gmat : numpy.ndarray
        geometry matrix with shape (#channels, #nodes)
    signal : numpy.ndarray
    gdg : numpy.ndarray
       :math:`\mathbf{T}^T \cdot \mathbf{T}` symmetrised geometry matrix, shape (#nodes, #nodes)
    gdsig : numpy.ndarray
        right side of regularisation scheme :math:`\mathbf{T}^T \cdot \mathbf{f}`
    """
    def __init__(self):
        super().__init__()
        self.last_chi = None
        self.logalpha = None
        self.gmat = None
        self.signal = None
        self.gdg = None
        self.gdsig = None

    @staticmethod
    def solve(a, b):
        r"""
        Finds solution of :math:`\mathbf{Ax}=\mathbf{b}` using scipy.sparse.linalg.spsolve
        """
        return spsolve(a, b)

    def test_regularization(self, logalpha, objective):
        """
        Function passed to minimisation function used for finding regularisation parameter value.

        Inverses signals using given regularisation parameter and computes chi2 test

        Stores pearson test result in attribute last_chi.

        Parameters
        ----------
        logalpha : float
            natural logarithm of regularisation parameter
        objective : numpy.ndarray
            a discretised objective functional also called regularisation or smoothing matrix, with shape (#nodes, #nodes)
            matrix created from smoothing matrix and results from previous MFR step

        Returns
        -------
        abs(chi2 - 1) : float
            1D Euclidean distance from ideal Pearson test result
        """
        alpha = 10 ** logalpha
        mod_mat = self.gdg + alpha * objective
        g = self.solve(mod_mat, self.gdsig)
        chi2 = self.pearson_test(g)
        self.last_chi = chi2
        return abs(chi2 - 1)

    def invert(self, signals, gmat, derivatives, w_factor=None, mfi_num=3, bounds=(-15, 0), iter_max=10, w_max=1,
               danis=0):
        """
        Inverses normalised signals using `mfi_num` Fisher Information cycles each with `iter_max` steps of regularisation parameter optimisation.

        See smoothing_mat documentation for more information about derivative matrix formats.

        Parameters
        ----------
        signals : numpy.ndarray
            error normalised signals on detector channels
        gmat : scipy.sparse.csr_matrix
            geometry matrix normalised by estimated errors
        derivatives : list
            list of tuples containing pairs of sparse derivatives matrices
        w_factor : numpy.ndarray, optional
            weight matrix multipliers with shape (#nodes, )
        mfi_num : float, optional
            number of MFR iterations
        bounds : tuple of two floats, optional
            exponent values for bounds of regularisation parameter alpha
        iter_max : float, optional
            maximum number of root finding iterations
        w_max : float, optional
            value used in weigh matrix for zero or negative nodes
        danis : float
            Determines anisotropy of derivative matrices, passed to sigmoid function

        Returns
        -------
        numpy.ndarray
            vector with nodes emissivities, shape (#nodes, )
        

        See Also
        --------
        smoothing_mat
        """
        ela = time.time()
        self.signal = signals
        self.gmat = gmat
        self.gdg = gmat.T.dot(gmat)
        self.gdsig = gmat.T.dot(signals)
        npix = gmat.shape[1]
        g = np.ones(npix)
        mfi_count = 0
        while mfi_count < mfi_num:
            w = 1 / g
            w[w < 0] = w_max
            w = sparse.diags(w)
            if w_factor is not None:
                w = w * sparse.diags(w_factor)
            objective = self.smoothing_mat(w, derivatives, danis)
            res = minimize_scalar(self.test_regularization,
                                  method='bounded',
                                  bounds=bounds,
                                  args=objective,
                                  options={'maxiter': iter_max},
                                  )
            self.logalpha = res.x
            m = self.gdg + 10 ** res.x * objective
            g = self.solve(m, self.gdsig)
            mfi_count += 1
        ela = time.time() - ela
        print('last chi^2 = {:.4f}, time: {:.2f} s'.format(self.last_chi, ela))
        return g

    def smoothing_mat(self, w, derivatives, danis):
        """
        Computes smoothing matrix from provided derivative matrices and weight factors determined by emissivity from
        previous iteration of minimisation of Fisher Information.

        Multiple pairs of derivatives matrices computed by different numerical scheme can be used. Each pair should
        contain derivatives in each locally orthogonal direction.

        Parameters
        ----------
        w : scipy.sparse.dia.dia_matrix
            (#nodes, #nodes) diagonal matrix with pixel weight factors
        derivatives : list of scipy.sparse.csc_matrix pairs
            contains sparse derivative matrices, each pair contains derivatives in both locally orthogonal coordinates
        danis : float
            anistropic factor, positive values make derivatives along first coordinate more significant

        Returns
        -------
        smooth
            smoothing matrix
        """
        s1 = 0
        s2 = 0
        for pair in derivatives:
            s1 = s1 + pair[0].T.dot(w).dot(pair[0])
            s2 = s2 + pair[1].T.dot(w).dot(pair[1])
        smooth = self.sigmoid(danis) * s1 + self.sigmoid(-danis) * s2
        return smooth

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def pearson_test(self, g):
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
        numpy.float64
        """
        retrofit = self.gmat.dot(g)
        misfit = retrofit - self.signal
        misfit_sq = np.power(misfit, 2)
        res = np.average(misfit_sq)
        return res

    def __call__(self, data, gmat, dmats, errors, mask=None, **kwargs):
        """
        Normalises signal and geometry matrix using estimated errors of measurement and
        then executes a sequence of MFR reconstructions.

        Parameters
        ----------
        data : numpy.ndarray
            input to be inverted with shape (#time slices, #channels)
        gmat : sparse.csr_matrix
            shape (#channels, #nodes)
        dmats : list
            list of tuples with derivative matrix pairs
        errors : int, float or np.ndarray
            Can have shapes (#channels, ), (#time slices,) or (#tslices, #channels)
        
        Returns
        -------
        res : np.ndarray
            tomographic reconstruction, with shape (#time slices, #nodes)
        chi : np.ndarray
            Pearson test values for final results, shape (#time slices, )
        """
        if np.ndim(data) == 1:  # flat data, assume one time slice
            data = data.reshape(1, -1)
        nslices = data.shape[0]
        nchnls = data.shape[1]
        nnodes = gmat.shape[1]

        if np.ndim(errors) == 1:
            if errors.size == nslices:
                errors = errors.reshape(-1, 1)
            elif errors.size == nchnls:
                errors = errors.reshape(1, -1)
            else:
                raise ValueError('Incompatible shape of errors {} with the data {}'.format(errors.shape, data.shape))
        
        errors = errors * np.ones_like(data)
        signal_nrm = data / errors

        res = np.empty((nslices, nnodes))
        chi = np.empty(nslices)

        for i in range(nslices):
            signal_np = signal_nrm[i, :]
            error_sp = sparse.diags(1/errors[i, :])
            gmat_nrm = error_sp.dot(gmat)
            res[i] = self.invert(signal_np, gmat_nrm, dmats, **kwargs)
            chi[i] = self.last_chi

        return res, chi


class CholmodMfr(Mfr):
    """
    Implementation of sparse cholesky decomposition as solve method for standard MFR algorithm.

    Uses scikit.sparse package imported in init to solve the regularised task.
    """
    def __init__(self):
        """
        Executes standard initialization and imports sksparse.cholmod
        """
        super().__init__()
        from sksparse.cholmod import cholesky
        self.cholesky = cholesky

    def solve(self, a, b):
        r"""
        Finds solution of :math:`\mathbf{Ax}=\mathbf{b}` using sksparse.cholmod.cholesky
        """
        factor = self.cholesky(a)
        return factor(b)
