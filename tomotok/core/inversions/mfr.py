# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Minimum Fisher Regularisation implemented based on articles by M. Anton and J. Mlynar

.. [MFR1] M. Anton et al., "X-ray tomography on the TCV tokamak.", Plasma Phys. Control. Fusion  38.11 (1996): 1849.

.. [MFR2] J. Mlynar et al., "Current research into applications of tomography for fusion diagnostics." J. Fusion Energy 38.3 (2019): 458-466
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
    """
    def __init__(self):
        super().__init__()
        self._gmat = None
        self._signal = None
        self._gdg = None
        self._gdsig = None

    @staticmethod
    def invert(a, b):
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
        g = self.invert(mod_mat, self._gdsig)
        chi2 = self._pearson_test(g)
        return abs(chi2 - 1)

    def solve(self, signals, gmat, derivatives, derivative_weights=None, w_factor=None, mfi_num=3, bounds=(-15, 0), iter_max=15, w_max=1,
              tolerance=0.05, zero_negative=False):
        """
        Solves the tomography problem for given normalised signals using 'mfi_num' Fisher Information cycles
        each with 'iter_max' steps of regularisation parameter optimisation.

        See regularisation_matrix documentation for more information about derivative matrix formats.

        Parameters
        ----------
        signals : numpy.ndarray
            error normalised signals on detector channels
        gmat : scipy.sparse.csr_matrix
            geometry matrix normalised by estimated errors, shape (#channels, #nodes)
        derivatives : list
            list of sparse derivative matrices used to create regularisation matrix
        derivative_weights : list of float
            list of weights for individual derivative matrices
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
        zero_negative : bool
            sets negative values in previous step result to zero when initializing MFI loop

        Returns
        -------
        g : numpy.ndarray
            vector with nodes emissivities, shape (#nodes, )
        stats : dict
            inversion statistics
                chi : list of float
                    Pearson test value of each optimization result
                iter_num : list of int
                    numbers of iterations taken in each MFI loop
                logalpha : list of float
                    final regularisation parameter logarithm of each MFI loop

        See Also
        --------
        regularisation_matrix
        """
        ela = time.time()
        self._signal = signals
        self._gmat = gmat
        self._gdg = gmat.T @ gmat
        self._gdsig = gmat.T @ signals
        g = np.ones(gmat.shape[1])
        mfi_count = 0
        iter_nums = []
        alphas = []
        chis = []
        while mfi_count < mfi_num:
            # MFI loop searching for ideal value of regularisation parameter
            w = 1 / g
            w[w < 0] = w_max
            w = sparse.diags(w)
            if w_factor is not None:
                w = w * sparse.diags(w_factor)
            if zero_negative:
                g[g < 0] = 0
            regularisation, stats = self.determine_regularisation(
                derivatives,
                w,
                derivative_weights,
                bounds,
                iter_max,
                tolerance
            )
            iter_nums.append(stats['iter_num'])
            alphas.append(stats['logalpha'])
            m = self._gdg + regularisation
            g = self.invert(m, self._gdsig)
            chi_sq = self._pearson_test(g)
            chis.append(chi_sq)
            mfi_count += 1
        # logalpha = res.x
        ela = time.time() - ela
        stats = dict(chi=chis, logalpha=alphas, iter_num=iter_nums, elapsed=ela)
        return g, stats

    def determine_regularisation(self, derivatives, w, derivative_weights, bounds, iter_max, tolerance):
        """
        Uses minimize scalar function from scipy to iteratively minimize chi square (Pearson test).
        """
        stats = dict()
        objective = self.regularisation_matrix(derivatives, w, derivative_weights)
        # TODO write custom optimisation routine to avoid recalculating optimal solution to get chi sq
        res = minimize_scalar(
            self._test_regularization,
            method='bounded',
            bounds=bounds,
            args=objective,
            options={'maxiter': iter_max, 'xatol': tolerance},
        )
        if res.status == 1:
            warn('Maximum number of iteration in MFI loop reached. Consider increasing iter_max.')
        stats['iter_num'] = res.nfev
        stats['logalpha'] = res.x
        regularisation = 10**res.x * objective
        return regularisation, stats

    def regularisation_matrix(self, derivatives, weights, derivative_weights=None):
        """
        Computes nonlinear regularisation matrix from provided derivative matrices and node weight factors 
        determined by emissivity from previous iteration of the inversion loop.
        
        Multiple derivative matrices can be used allowing to combine matrices computed by 
        different numerical schemes. 
        
        Each matrix can have different weight assigned to introduce anisotropy.

        Parameters
        ----------
        derivatives : list of scipy sparse matrix
            a list of derivative matrices, with shape (# nodes, # nodes) 
        weights : scipy.sparse.dia_matrix
            node weight factors, (#nodes, #nodes) 
        weight_coefficients : list of float
            allows to specify anisotropy by assign weights for each matrix

        Returns
        -------
        scipy.sparse.csc_matrix
        """
        if isinstance(derivatives, sparse.spmatrix):
            derivatives = [derivatives]
        if derivative_weights is None:
            derivative_weights = [1] * len(derivatives)
        total = sum(derivative_weights)
        regularisation = sparse.csr_matrix(derivatives[0].shape)
        for dw, dmat in zip(derivative_weights, derivatives):
            regularisation += dw / total * dmat.T @ weights @ dmat
        return regularisation

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
        self._chisq = np.average(misfit_sq)
        return self._chisq

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
            keywords passed to solve method
        
        Returns
        -------
        res : numpy.ndarray
            tomographic reconstruction, with shape (#time slices, #nodes)
        stats_list : list of dicts
            contains dicts with inversion statistics returned by solve method
        
        See also
        --------
        solve
        """
        data_ndim = np.ndim(data)
        if data_ndim == 0:
            raise ValueError('Data must be at least an 1D array of values.')
        elif data_ndim == 1:  # flat data, assume one time slice
            data = data.reshape(1, -1)
        elif data_ndim > 2:
            raise ValueError('Data array has too many dimension: {}. Max is 2.'.format(data_ndim))

        if len(gmat.shape) != 2:
            raise ValueError('Gmat must be a 2D array or matrix')

        nslices = data.shape[0]
        nchnls = data.shape[1]
        if nchnls != gmat.shape[0]:
            raise ValueError('Different number of channels in data and gmat')
        nnodes = gmat.shape[1]

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
            res[i], stats = self.solve(signal_np, gmat_nrm, derivatives, **kwargs)
            stats_list.append(stats)

        return res, stats_list


class CholmodMfr(Mfr):
    """
    Uses sparse cholesky decomposition for solving the parameter optimisation task in MFI loop.
    Requires scikit sparse to be installed in order to initialize properly.

    Uses sksparse.cholmod.cholesky to solve the regularised task in parameter optimisation.
    """
    def __init__(self):
        """
        Executes standard initialization and imports sksparse.cholmod
        """
        super().__init__()
        from sksparse.cholmod import cholesky
        self.cholesky = cholesky

    def invert(self, a, b):
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
