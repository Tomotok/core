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
from pathlib import Path
from typing import Union

import h5py
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import linalg as sp_linalg


from tomotok.core.tools.hdf import sparse_to_hdf, hdf_to_sparse


class Bob(object):
    """
    BiOrthogonal Basis decomposition

    Attributes
    ----------
    basis : scipy.sparse.spmatrix
        :math:`\mathbf{b}_i` basis vectors of reconstruction plane
    basis_inv : scipy.sparse.spmatrix
        inverse of basis matrix, used for transformation to node basis
    dec_mat : scipy.sparse.csr_matrix
        :math:`\hat{\mathbf{e}}_i` decomposed matrix used to transform image into reconstruction plane
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
        self.adjoint_basis = dec_mat
        self.norms = None
        return

    def decompose(self, gmat: sparse.csr_matrix, basis: sparse.csr_matrix, reg_factor: float = 0, solver_kw: dict = None):
        """
        Decomposes the geometry matrix using basis vectors

        Parameters
        ----------
        gmat : scipy.sparse.csr_matrix
            geometry/contribution matrix
        basis : sparse matrix
            matrix with decomposition basis vectors
        reg_factor : float, optional
            regularisation factor passed to cholesky decomposition
            determines weight of regularisation by identity matrix relatively to arbitrary matrix maximum value
        solver_kw : dict
            keyword parameters passed to the compute_coefficients method
        
        See Also
        --------
        compute_coefficients : method handling computation of coefficients to see supported solver keywords
        """
        solver_kw = solver_kw or {}
        los_num = gmat.shape[0]
        node_num = gmat.shape[1]
        if los_num < node_num:
            warnings.warn('Biorthogonal algorithm requires more lines of sights than nodes in reconstruction plane to run reliably')
        self.basis = basis
        self.basis_inv = sp_linalg.inv(self.basis)
        projection_basis = gmat @ self.basis  # A with columns of a_i, projections of decomposition basis
        ata = (projection_basis.T @ projection_basis)  # <A^T|A> symmetrized projections
        if reg_factor:
            ata = ata + ata.max() * reg_factor * sparse.eye(*ata.shape, format='csc')
        c = self.compute_coordinates(ata, **solver_kw)  # coordinate matrix
        self.adjoint_basis = projection_basis @ c  # \hat{b}_i previously known as xi
        return

    def compute_coordinates(self, a: sparse.csc_matrix, check_finite=False) -> sparse.csr_matrix:
        """
        Computes coordinate matrix using cho_factor and cho_solve from scipy.linalg.

        Parameters
        ----------
        a : scipy.sparse.csr_matrix
            square and positive definite matrix
        check_finite : bool, optional
            toggles checking elements in a by cho_factor, by default False
        """
        if isinstance(a, sparse.spmatrix):
            a = a.toarray()
        factor = cho_factor(a, check_finite=check_finite)
        b = np.eye(a.shape[0])
        c = cho_solve(factor, b, check_finite=False)  # no need to check again
        return sparse.csr_matrix(c)

    def __call__(self, data: np.ndarray, gmat: sparse.csr_matrix = None, thresholding=None, **kw) -> np.ndarray:
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
        # FIXME: transpose data and res so that each time slice is a column
        if thresholding is not None:
            warnings.warn('Thresholding not implemented to call method. Ignoring.')
        if self.adjoint_basis is None:
            if gmat is None:
                raise ValueError('Gmat must be provided for decomposition')
            else:
                self.decompose(gmat)
        coeffs = self.adjoint_basis.T @ data  # coordinates in decomposition basis
        # TODO: no need to normalise here, none of the bases is
        # if self.norms is None:
        #     self.normalise()
        # coeffs = coeffs * self.norms
        # FIXME: work only for simple basis -> coeffs @ (basis) in dec. basis coords
        # res = self.basis @ coeffs  # result in node basis
        res = coeffs.T @ self.basis_inv
        return res

    def save_decomposition(self, floc: Union[str, Path], description: str = '') -> None:
        """
        Saves decomposition matrix and basis to hdf file. Norms are also included if calculated.

        Parameters
        ----------
        floc : str or pathlib.Path
            file location with name
        description : str, optional
            short user description for file identification
        """
        if self.adjoint_basis is None:
            raise ValueError('Can not save decomposition before it is calculated.')
        floc = str(floc)
        with h5py.File(floc, 'w') as f:
            f.attrs['version'] = '0.1'
            f.attrs['description'] = description
            dec_mat = f.create_group('decomposed_matrix')
            sparse_to_hdf(self.adjoint_basis, dec_mat)
            basis = f.create_group('basis')
            sparse_to_hdf(self.basis, basis)
            if self.norms is not None:
                f.create_dataset('norms', data=self.norms)
    
    def load_decomposition(self, floc: Union[str, Path]) -> None:
        """
        Loads decomposed matrix and basis from an HDF file.

        Norms are loaded only if available in file.

        Parameters
        ----------
        floc : str or pathlib.Path
            location of hdf file with saved decomposition
        """
        with h5py.File(floc, 'r') as f:
            self.adjoint_basis = hdf_to_sparse(f['decomposed_matrix'])
            self.basis = hdf_to_sparse(f['basis'])
            try:
                self.norms = f['norms'][:]
            except KeyError:
                self.norms = None
        self.basis_inv = sp_linalg.inv(self.basis)
        return

    def normalise(self, precision: float = 1e-6) -> None:
        """
        Computes normalisation factors for decomposition matrix.

        Parameters
        ----------
        precision : float, optional
            neglects decomposition matrix rows with lower norm, by default 1e-6
        """
        adjoint_basis = self.adjoint_basis
        kappa = sparse.linalg.norm(adjoint_basis, axis=0)
        idx = kappa > precision
        norms = np.zeros(kappa.size)
        norms[idx] = (1 / kappa[idx])
        self.norms = norms[:, None]  # change to expected shape
        return

    def thresholding(self, image, c: int, precision: float = 1e-6, conv: float = 1e-9) -> np.ndarray:
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
        if self.adjoint_basis is None:
            raise RuntimeError('Decomposition must be computed prior to thresholding.')
        if self.norms is None:
            self.normalise(precision)

        # calculate plane basis coefficients
        coeffs = self.adjoint_basis.T @ image

        # thresholding loop
        a = np.abs(coeffs * self.norms)  # normalised coefficients
        threshold_2 = np.sqrt(c**2 / a.size * a.T @ a)
        threshold_1 = 0
        while np.abs(threshold_1-threshold_2) >= conv:
            threshold_1 = threshold_2
            a_temp = a[a <= threshold_1]
            threshold_2 = np.sqrt(c**2 / a_temp.size * a_temp.T @ a_temp)

        # remove plane basis with contributions below threshold and transform to nodes
        coeffs[a < threshold_2] = 0
        out = self.basis @ coeffs
        return out


class SparseBob(Bob):
    """
    Biorthogonal Basis Decomposition optimized for sparse matrices using inverse matrix calculation.
    """

    def compute_coordinates(self, a: sparse.csc_matrix) -> sparse.csr_matrix:
        try:
            c = sparse.linalg.inv(a)
        except RuntimeError:
            raise ValueError('Singular symmetrized matrix factor. Try increasing regularisation factor.')
        return c
