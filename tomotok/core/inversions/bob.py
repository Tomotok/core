# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Contains inversion class for Biorthogonal Basis Decomposition Algorithm proposed by J. Cavlier. 
It is a simplified form of wavelet-vaguelette decomposition algorithm by R. Nguyen van Yen

.. [BOB1] Jordan Cavalier et al., Nucl. Fusion 59 (2019): 056025
.. [BOB2] R. Nguyen van Yen et al., Nucl. Fusion 52 (2011): 013005
"""
import warnings

import h5py
import numpy as np
import scipy.sparse as sparse

from tomotok.core.tools.hdf import sparse_to_hdf, hdf_to_sparse


class Bob(object):
    """
    BiOrthogonal Basis decomposition

    Attributes
    ----------
    basis : scipy.sparse.dia_matrix
        :math:`\mathbf{b}_i` basis vectors of reconstruction plane
    dec_mat : scipy.sparse.csr_matrix
        :math:`\hat{\mathbf{e}}_i` decomposed matrix used to transform image into reconstruction plane
    dec_mat_normed : scipy.sparse.csr_matrix
        normalised decomposed matrix
    norms : numpy.ndarray
        node norms used in thresholding
    """

    def __init__(self, dec_mat=None, basis=None):
        """
        Parameters
        ----------
        dec_mat : scipy.sparse.csr_matrix, optional
            previously decomposed matrix, avoids recomputation of decomposition when provided
        basis : array_like, optional
            A set of basis vectors used for decomposition
        """
        # TODO one parameter holding both dec_mat and basis?
        super().__init__()
        self.basis = basis
        self.dec_mat = dec_mat
        self.norms = None
        self.dec_mat_normed = None
        return

    def decompose(self, gmat, basis, reg_factor=0, solver_kw: dict=None):
        """
        Decomposes the geometry matrix using basis vectors

        Parameters
        ----------
        gmat : scipy.sparse.csr_matrix
            geometry/contribution matrix
        basis : sparse matrix
            matrix with basis vectors
        reg_factor : float, optional
            regularisation factor passed to cholesky decomposition
            determines weight of regularisation by identity matrix relatively to arbitrary matrix maximum value
        solver_kw : dict
            keyword parameters passed to the solver function
        """
        solver_kw = solver_kw or {}
        if gmat.shape[0] < gmat.shape[1]:
            warnings.warn('Biorthogonal algorithm requires more lines of sights than nodes in reconstruction plane to run reliably')
        self.basis = basis
        image_base = gmat.dot(self.basis)  # e_i previously known as chi, gmat in basis
        a = image_base.T.dot(image_base).toarray()  # symmetrized geometry matrix in basis
        if reg_factor:
            a = a + a.max() * reg_factor * np.eye(*a.shape)
        b = np.eye(gmat.shape[1])
        res = np.linalg.lstsq(a, b, **solver_kw)
        c = sparse.csr_matrix(res[0])  # coefficient matrix
        self.dec_mat = image_base.dot(c)  # \hat{e}_i previously known as xi, decomposed matrix
        return

    def __call__(self, data, gmat=None, thresholding=None, **kw):
        """
        Decomposes geometry matrix and projects images

        Parameters
        ----------
        data : numpy.ndarray
            contains signals with shape (# channels, # time slices)
        gmat : scipy.sparse.csr_matrix
            geometry matrix
        thresholding : float, optional
            not implemented
        
        Returns
        -------
        numpy.ndarray
            inversion results with shape (# nodes, # time slices)
        """
        # TODO transpose data and res?
        if thresholding is not None:
            warnings.warn('Thresholding not implemented to call method. Ignoring.')
        if self.dec_mat is None:
            if gmat is None:
                raise ValueError('Gmat must be provided for decomposition')
            else:
                self.decompose(gmat)
        coeffs = self.dec_mat.T.dot(data)  # coordinates in reconstruction basis
        res = self.basis.dot(coeffs)  # result in node basis
        return res

    def save_decomposition(self, floc, description=''):
        """
        Saves decomposition matrix and basis to hdf file. Norms are also included if calculated.

        Parameters
        ----------
        floc : str or pathlib.Path
            file location with name
        description : str, optional
            short user description for file identification
        """
        if self.dec_mat is None:
            raise ValueError('Can not save decomposition before it is calculated.')
        floc = str(floc)
        with h5py.File(floc, 'w') as f:
            f.attrs['version'] = '0.1'
            f.attrs['description'] = description
            dec_mat = f.create_group('decomposed_matrix')
            sparse_to_hdf(self.dec_mat, dec_mat)
            basis = f.create_group('basis')
            sparse_to_hdf(self.basis, basis)
            if self.norms is not None:
                f.create_dataset('norms', data=self.norms)
    
    def load_decomposition(self, floc):
        """

        Parameters
        ----------
        floc : str or pathlib.Path
            location of hdf file with saved decomposition
        """
        with h5py.File(floc, 'r') as f:
            self.dec_mat = hdf_to_sparse(f['decomposed_matrix'])
            self.basis = hdf_to_sparse(f['basis'])
            try:
                self.norms = f['norms'][:]
            except KeyError:
                self.norms = None
        return

    def normalise(self, precision=1e-6):
        """
        Computes normalised decomposition matrix.

        Parameters
        ----------
        precision : float, optional
            neglects decomposition matrix rows with lower norm, by default 1e-6
        """
        image_base_adj = self.dec_mat
        kappa = sparse.linalg.norm(image_base_adj, axis=0)
        idx = kappa > precision
        norms = np.zeros(kappa.size)
        norms[idx] = (1 / kappa[idx])
        # xi_norm
        self.dec_mat_normed = image_base_adj.multiply(sparse.csr_matrix(norms))
        self.norms = norms[:, None]  # change to expected shape
    
    def _normalise_wo_mat(self, precision=1e-6):
        """
        Computes norms for reconstruction nodes.

        Parameters
        ----------
        precision : float, optional
            neglects decomposition matrix rows with lower norm, by default 1e-6
        """
        image_base_adj = self.dec_mat
        kappa = sparse.linalg.norm(image_base_adj, axis=0)
        idx = kappa > precision
        norms = np.zeros(kappa.size)
        norms[idx] = (1 / kappa[idx])
        self.norms = norms[:, None]  # change to expected shape

    def thresholding(self, image, c: int, precision: float=1e-6, conv: float=1e-9):
        """
        Applies thresholding method to provided image.

        Parameters
        ----------
        image : numpy.ndarray
            flattened image with shape (#pixels, 1)
        c : int
            thresholding sensitivity constant
        precision : float, optional
            normalisation precision, by default 1e-6
        conv : float, optional
            thresholding convergence limit

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        RuntimeError
            If thresholding is called before decomposition of geometry matrix
        """
        if self.dec_mat is None:
            raise RuntimeError('Decomposition must be computed prior to thresholding.')
        if self.norms is None:
            self.normalise(precision)

        # calculate plane basis coefficients
        coeffs = self.dec_mat.T @ image

        # thresholding loop
        a = np.abs(coeffs * self.norms)  # normalised coefficients
        threshold_2 = np.sqrt(c**2 / a.size * a.T.dot(a))
        threshold_1 = 0
        while np.abs(threshold_1-threshold_2) >= conv:
            threshold_1 = threshold_2
            a_temp = a[a <= threshold_1]
            threshold_2 = np.sqrt(c**2 / a_temp.size * a_temp.T.dot(a_temp))

        # remove plane basis with contributions below threshold and transform to nodes
        coeffs[a < threshold_2] = 0
        out = self.basis @ coeffs
        return out


class SimpleBob(Bob):
    """
    Automatically creates simple one node basis as an sparse identity matrix

    .. deprecated:: 1.1
    """

    def __init__(self, dec_mat=None, basis=None):
        warnings.warn('SimpleBob is deprecated since v1.1', DeprecationWarning)
        super().__init__(dec_mat, basis)
    
    def decompose(self, gmat, basis=None):
        if basis is not None:
            warnings.warn('Ignoring basis input')
        basis = sparse.eye(gmat.shape[1])
        return super().decompose(gmat, basis)


class SparseBob(Bob):
    """
    Biorthogonal Basis Decomposition optimized for sparse matrices using inverse matrix calculation.
    """

    def decompose(self, gmat, basis, reg_factor=0, solver_kw=None):
        if solver_kw is not None:
            raise TypeError('scipy.sparse.linalg.inv does not take any keywords')
        if gmat.shape[0] < gmat.shape[1]:
            warnings.warn('Biorthogonal algorithm requires more '
            'lines of sights than nodes in reconstruction plane to run reliably')
        self.basis = basis
        image_base = gmat.dot(self.basis)  # chi
        a = image_base.T.dot(image_base)
        if reg_factor:
            a = a + a.max() * reg_factor * sparse.eye(*a.shape)
        try:
            c = sparse.linalg.inv(a)
        except RuntimeError:
            raise ValueError('Singular symmetrized matrix factor. Try increasing regularisation factor.')
        self.dec_mat = image_base.dot(c)  # xi
        return


class CholmodBob(Bob):
    """
    Decomposition optimized for sparse matrices using Cholesky decomposition

    Uses sksparse.cholmod.cholesky to solve the decomposition
    Requires positive definite symmetrized geometry matrix in reconstruction plane basis.
    """

    def decompose(self, gmat, basis, reg_factor=1e-3, solver_kw=None):
        """
        Decomposes geometry matrix using Cholesky decomposition and projects images

        Parameters
        ----------
        gmat : scipy.sparse.csr_matrix
            geometry/contribution matrix
        basis : sparse matrix
            matrix with basis vectors
        reg_factor : float, optional
            regularisation factor passed to cholesky decomposition
            determines weight of regularisation by identity matrix relatively to arbitrary matrix maximum value
        solver_kw : dict
            keyword parameters passed to the solver function
        """
        solver_kw = solver_kw or {}
        from sksparse.cholmod import cholesky, CholmodNotPositiveDefiniteError
        if gmat.shape[0] < gmat.shape[1]:
            warnings.warn('Biorthogonal algorithm can be prone to failure if there are more '
            'lines of sights than nodes in reconstruction plane')
        self.basis = basis
        image_base = gmat.dot(self.basis)  # chi
        a = image_base.T.dot(image_base)
        try:
            factor = cholesky(a, a.max()*reg_factor, **solver_kw)
        except CholmodNotPositiveDefiniteError:
            raise ValueError('Symmetrized matrix was not positive definite. Try increasing regularisation factor.')
        b = sparse.csc_matrix(np.eye(gmat.shape[1]))
        c = factor(b)
        self.dec_mat = image_base.dot(c)  # xi
        return
