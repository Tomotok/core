# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Contains inversion class for Biorthogonal Basis Decomposition Algorithm [1]_ derrived from wavelet-vaguelette decomposition [2]_

.. [1] Jordan Cavalier et al 2019 Nucl. Fusion 59 056025
.. [2] R. Nguyen van yen et al 2011 Nucl. Fusion 52 013005
"""
import warnings

import numpy as np
import scipy.sparse as sparse


class Bob(object):
    """
    BiOrthogonal Basis decomposition

    Attributes
    ----------
    basis : array_like
        :math:`\mathbf{b}_i` basis vectors of reconstruction plane
    dec_mat : scipy.sparse.csr_matrix
        :math:`\hat{\mathbf{e}}_i` decomposed matrix used to transform image into reconstruction plane
    dec_mat_normed : scipy.sparse.csr_matrix
        normalised decomposed matrix
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
        b = np.eye(gmat.shape[1], **solver_kw)
        res = np.linalg.lstsq(a, b)
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
            if provided, result is processed using thresholding, not performed by default 
        """
        if self.dec_mat is None:
            if gmat is None:
                raise ValueError('Gmat must be provided for decomposition')
            else:
                self.decompose(gmat)
        coeffs = self.dec_mat.T.dot(data)  # coordinates in reconstruction basis
        res = self.basis.dot(coeffs)  # result in node basis
        return res

    # TODO remove? remove normalised parameter?
    def save_decomposition(self, floc, normalised=False):
        """
        Saves 

        Parameters
        ----------
        floc : str or pathlib.Path
            file location with name
        normalised : bool, optional
            selects whether to save normalised matrix, by default False
        """
        floc = str(floc)
        if normalised:
            sparse.save_npz(floc, self.dec_mat_normed, compressed=True)
        else:
            sparse.save_npz(floc, self.dec_mat, compressed=True)

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
        xi_norm = image_base_adj.multiply(sparse.csr_matrix(norms))
        self.dec_mat_normed = xi_norm

    # TODO replace image by reconstruction?
    # TODO return mask only?
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
            If tresholding is called before decomposition of geometry matrix
        """
        if self.dec_mat is None:
            raise RuntimeError('Decomposition must be computed prior to thresholding.')
        if self.dec_mat_normed is None:
            self.normalise(precision)
        temp = sparse.csr_matrix(image)
        a = np.abs(self.dec_mat_normed.T.dot(temp).toarray())

        threshold_2 = np.sqrt(c**2 / a.size * a.T.dot(a))
        threshold_1 = 0
        while np.abs(threshold_1-threshold_2) >= conv:
            threshold_1 = threshold_2
            a_temp = a[a <= threshold_1]
            threshold_2 = np.sqrt(c**2 / a_temp.size * a_temp.T.dot(a_temp))

        out = self.dec_mat.T.dot(image)
        out = self.basis.dot(out)
        out[a < threshold_2] = 0
        return out


class SimpleBob(Bob):
    """
    Automatically creates simple one node basis as an sparse identity matrix

    .. deprecated:: 1.1
    """

    def __init__(self, dec_mat=None, basis=None):
        warnings.warn('SimpleBob is depracated since v1.1', DeprecationWarning)
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

    def decompose(self, gmat, basis, reg_factor=0):
        if gmat.shape[0] < gmat.shape[1]:
            warnings.warn('Biorthogonal algorithm requires more '
            'lines of sights than nodes in reconstruction plane to run reliably')
        self.basis = basis
        chi = gmat.dot(self.basis)
        a = chi.T.dot(chi)
        if reg_factor:
            a = a + a.max() * reg_factor * sparse.eye(*a.shape)
        try:
            c = sparse.linalg.inv(a)
        except RuntimeError:
            raise ValueError('Singular symmetrized matrix factor. Try increasing regularisation factor.')
        self.dec_mat = chi.dot(c)  # xi
        return


class CholmodBob(Bob):
    """
    Decomposition optimized for sparse matrices using Cholesky decomposition

    Uses sksparse.cholmod.cholesky to solve the decomposition
    Requires positive definite symmetrized geometry matrix in reconstruction plane basis.
    """

    def decompose(self, gmat, basis, reg_factor=1e-3, solver_kw: dict=None):
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
