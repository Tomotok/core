# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Fixed parameter Tikhonov scheme with MFR like regularisation matrix
"""
import time
from warnings import warn

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve


class Fixt(object):
    r"""
    Inverses provided data using fixed parameter value Minimum Fisher Regularization scheme.

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

    def invert(self, signals, gmat, derivatives, parameters, w_factor=None, mfi_num=3, w_max=1,
               aniso=0):
        """
        Inverses normalised signals using fixed value of regularisation parameter.

        See regularisation_matrix documentation for more information about derivative matrix formats.

        Parameters
        ----------
        signals : numpy.ndarray
            error normalised signals on detector channels
        gmat : scipy.sparse.csr_matrix
            geometry matrix normalised by estimated errors
        derivatives : list
            list of tuples containing pairs of sparse derivatives matrices
        parameters : float or list of float
            regularisation parameter value(s), list length must match mfi_num
        w_factor : numpy.ndarray, optional
            weight matrix multipliers with shape (#nodes, )
        w_max : float, optional
            value used in weigh matrix for zero or negative nodes
        aniso : float, optional
            Determines anisotropy of derivative matrices

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
        if isinstance(parameters, float):
            parameters = mfi_num * [parameters]
        elif len(parameters) != mfi_num:
            msg = 'Different number of regularisation parameters: {} and MFI loops: {}.'
            raise ValueError(msg.format(len(parameters), mfi_num))
        ela = time.time()
        self._signal = signals
        self._gmat = gmat
        self._gdg = gmat.T.dot(gmat)
        self._gdsig = gmat.T.dot(signals)
        npix = gmat.shape[1]
        g = np.ones(npix)
        chis = []
        mfi_counter = 0
        while mfi_counter < mfi_num:
            w = 1 / g
            w[w < 0] = w_max
            w = sparse.diags(w)
            if w_factor is not None:
                w = w * sparse.diags(w_factor)
            objective = self.regularisation_matrix(derivatives, w, aniso)
            m = self._gdg + parameters[mfi_counter] * objective
            g = self.solve(m, self._gdsig)
            mfi_counter += 1
            last_chi = self._pearson_test(g)
            chis.append(last_chi)
        ela = time.time() - ela
        print('last chi^2 = {:.4f}, time: {:.2f} s'.format(last_chi, ela))
        stats = dict(chi=chis, parameter=parameters)
        return g, stats

    def smoothing_mat(self, w, derivatives, danis):
        """
        .. deprecated :: 1.1
            Use :meth:`regularisation_matrix`  
        """
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

    def __call__(self, data, gmat, derivatives, errors, parameters, **kwargs):
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
        parameters : float or list of float
            regularisation parameter value(s), list length must match mfi_num
        **kwargs : dict
            keywords passed to invert method
        
        Returns
        -------
        res : numpy.ndarray
            tomographic reconstruction, with shape (#time slices, #nodes)
        stats_list : list of dicts
            contains dicts with inversion statistics returned by invert method
        
        See also
        --------
        invert
        """
        try:
            kwargs['aniso'] = kwargs.pop('danis')
            warn('Parameter `danis` renamed to `aniso` since v1.1', DeprecationWarning)
        except KeyError:
            pass

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
            res[i], stats = self.invert(signal_np, gmat_nrm, derivatives, parameters, **kwargs)
            stats_list.append(stats)

        return res, stats_list


class CholmodFixt(Fixt):
    """
    Cholmod version of fixed parameter MFR
    Requires scikit sparse to be installed in order to initialize properly.

    Uses sksparse.cholmod.cholesky to solve the inversion problem
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
