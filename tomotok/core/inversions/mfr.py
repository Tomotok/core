# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Minimun Fisher Regularisation implemented based on M. Anton and J. Mlynar

M. Anton et al., "X-ray tomography on the TCV tokamak.", Plasma Phys. Control. Fusion  38.11 (1996): 1849.

J. Mlynar et al., "Current research into applications of tomography for fusion diagnostics." J. Fusion Energy 38.3 (2019): 458-466
"""
import time
from warnings import warn

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
        self._gmat = None
        self._signal = None
        self._gdg = None
        self._gdsig = None

    @staticmethod
    def solve(a, b):
        r"""
        Finds solution of :math:`\mathbf{Ax}=\mathbf{b}` using scipy.sparse.linalg.spsolve
        """
        return spsolve(a, b)

    def _test_regularization(self, logalpha, objective):
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
        mod_mat = self._gdg + alpha * objective
        g = self.solve(mod_mat, self._gdsig)
        chi2 = self._pearson_test(g)
        return abs(chi2 - 1)

    def invert(self, signals, gmat, derivatives, w_factor=None, mfi_num=3, bounds=(-15, 0), iter_max=15, w_max=1,
               aniso=0, tolerance=0.05):
        """
        Inverses normalised signals using `mfi_num` Fisher Information cycles each with 
        `iter_max` steps of regularisation parameter optimisation.

        See regularisation_matrix documentation for more information about derivative matrix formats.

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
        aniso : float, optional
            Determines anisotropy of derivative matrices
        tolerance : float, optional
            End MFI loop if residuum reaches interval (1-tolerance; 1+tolerance), by default 0.05

        Returns
        -------
        g : numpy.ndarray
            vector with nodes emissivities, shape (#nodes, )
        stats : dict
            inversion statistics

                chi : float
                    Pearson test value of final result
                niter : list of int
                    numbers of iterations taken in each MFI loop
                logalpha : float
                    logarithm of final regularisation parameter

        See Also
        --------
        regularisation_matrix
        """
        ela = time.time()
        self._signal = signals
        self._gmat = gmat
        self._gdg = gmat.T.dot(gmat)
        self._gdsig = gmat.T.dot(signals)
        npix = gmat.shape[1]
        g = np.ones(npix)
        mfi_count = 0
        niter = []
        while mfi_count < mfi_num:
            # MFI loop searching for ideal value of regularisation parameter
            w = 1 / g
            w[w < 0] = w_max
            w = sparse.diags(w)
            if w_factor is not None:
                w = w * sparse.diags(w_factor)
            objective = self.regularisation_matrix(derivatives, w, aniso)
            res = minimize_scalar(self._test_regularization,
                                  method='bounded',
                                  bounds=bounds,
                                  args=objective,
                                  options={'maxiter': iter_max, 'xatol': tolerance},
                                  )
            if res.status == 1:
                warn('Maximum number of iteration in MFI loop reached. Consider increasing iter_max.')
            niter.append(res.nfev)
            # TODO write custom optimisation routine to avoid recalculating optimal solution?
            m = self._gdg + 10 ** res.x * objective
            g = self.solve(m, self._gdsig)
            mfi_count += 1
        logalpha = res.x
        last_chi = self._pearson_test(g)
        ela = time.time() - ela
        print('last chi^2 = {:.4f}, time: {:.2f} s'.format(last_chi, ela))
        stats = dict(chi=last_chi, logalpha=logalpha, niter=niter)
        return g, stats

    def smoothing_mat(self, w, derivatives, danis):
        warn('Smoothing_mat deprecated by regularisation_matrix method since v1.1.', DeprecationWarning)
        self.regularisation_matrix(derivatives, w, danis)

    def regularisation_matrix(self, derivatives, w, aniso=0):
        """
        Computes nonlinear regularisation matrix from provided derivative matrices and node weight factors 
        determined by emissivity from previous iteration of the inversion loop.
        
        Anisotropic coefficients are computed using sigmoid function.

        Multiple pairs of derivatives matrices computed by different numerical scheme can be used. Each pair should
        contain derivatives in each locally orthogonal direction.

        Parameters
        ----------
        w : scipy.sparse.dia.dia_matrix
            (#nodes, #nodes) diagonal matrix with pixel weight factors
        derivatives : list of scipy.sparse.csc_matrix pairs
            contains sparse derivative matrices, each pair contains derivatives in both locally orthogonal coordinates
        aniso : float
            anistropic factor, positive values make derivatives along first coordinate more significant

        Returns
        -------
        scipy.sparse.csc_matrix
        """
        # sigmoid function
        w1 = 1 / (1 + np.exp(-aniso))
        w2 = 1 / (1 + np.exp(aniso))
        h1 = 0
        h2 = 0
        for pair in derivatives:
            h1 = h1 + pair[0].T.dot(w).dot(pair[0])
            h2 = h2 + pair[1].T.dot(w).dot(pair[1])
        smooth = w1 * h1 + w2 * h2
        return smooth

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
        retrofit = self._gmat.dot(g)
        misfit = retrofit - self._signal
        misfit_sq = np.power(misfit, 2)
        res = np.average(misfit_sq)
        return res

    def __call__(self, data, gmat, derivatives, errors, **kwargs):
        """
        Normalises signal and geometry matrix using estimated errors of measurement and
        then executes a sequence of MFR reconstructions.

        Parameters
        ----------
        data : numpy.ndarray
            input to be inverted with shape (#channels,), (#time slices, #channels)
        gmat : sparse.csr_matrix
            shape (#channels, #nodes)
        derivatives : list
            list of tuples with derivative matrix pairs
        errors : int, float or np.ndarray
            Can have shapes (#channels, ), (#time slices,) or (#time slices, #channels)
        **kwargs : dict
            keywords passed to invert method
        
        Returns
        -------
        res : numpy.ndarray
            tomographic reconstruction, with shape (#time slices, #nodes)
        stats_list : list of dicts
            contains dicts with inversion statistics returned by invert method
        """
        try:
            kwargs['aniso'] = kwargs.pop('danis')
            warn('Parameter `danis` renamed to `aniso` since v1.1', DeprecationWarning)
        except KeyError:
            pass

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

        signal_nrm = data / errors

        res = np.empty((nslices, nnodes))
        stats_list = []

        for i in range(nslices):
            signal_np = signal_nrm[i, :]
            error_sp = sparse.diags(1/errors[i, :])
            gmat_nrm = error_sp.dot(gmat)
            res[i], stats = self.invert(signal_np, gmat_nrm, derivatives, **kwargs)
            stats_list.append(stats)

        return res, stats_list


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

        Parameters
        ----------
        a : scipy.sparse.csr_matrix
            square and positive definite matrix
        b : array_like
            right hand side vector
        """
        factor = self.cholesky(a)
        return factor(b)
