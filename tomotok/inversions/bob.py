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

import h5py
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg

from .base import Inversion, Solver, CholeskySolver
from tomotok.tools.hdf import sparse_to_hdf, hdf_to_sparse


class Bob(Inversion):
    r"""
    BiOrthogonal Basis decomposition

    Attributes
    ----------
    basis : scipy.sparse.spmatrix
        :math:`\mathbf{b}_i` basis vectors of reconstruction plane
    basis_inv : scipy.sparse.spmatrix
        inverse of basis matrix, used for transformation to node basis
    decomposed_matrix : scipy.sparse.csr_matrix
        :math:`\hat{\mathbf{e}}_i` decomposed matrix used to transform image into reconstruction plane
    norms : numpy.ndarray
        node norms used in thresholding
    """
    def __init__(
        self,
        engine: Solver | None = None,
        decomposed_matrix: sparse.sparray | None = None,
        basis: sparse.sparray | None = None,
    ):
        """
        Parameters
        ----------
        engine : Engine, optional
            engine for solving linear systems in decomposition, by default CholeskyEngine
        decomposed_matrix : scipy.sparse.sparray, optional
            previously decomposed matrix, avoids recomputation of decomposition when provided
        basis : scipy.sparse.sparray, optional
            A set of basis vectors used for decomposition
        """
        engine = engine or CholeskySolver()
        super().__init__(solver=engine)
        self._basis = basis
        self._adjoint_basis = decomposed_matrix
        self._norms: np.ndarray | None = None

    # @property
    # def basis(self):
    #     return self._basis

    # @basis.setter
    # def basis(self, value):
    #     if value is not None:
    #         basis = sparse.csc_array(value)
    #     self._basis = basis

    def decompose(
        self,
        gmat: sparse.csr_array | sparse.csc_array,
        basis: sparse.csc_array,
        reg_factor: float = 0,
    ):
        """
        Decomposes the geometry matrix using basis vectors

        Parameters
        ----------
        gmat : scipy.sparse.csr_array or scipy.sparse.csc_array
            geometry/contribution matrix
        basis : sparse array
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
        los_num = gmat.shape[0]
        node_num = gmat.shape[1]
        if los_num < node_num:
            warnings.warn(
                'Biorthogonal algorithm requires more lines of sights than nodes in reconstruction plane to run reliably',
                RuntimeWarning,
                stacklevel=2,
            )
        self._basis = basis
        self.basis_inv = sp_linalg.inv(self._basis)
        projection_basis = gmat @ self._basis  # A with columns of a_i, projections of decomposition basis
        ata = (projection_basis.T @ projection_basis)  # <A^T|A> symmetrized projections
        if reg_factor:
            ata = ata + ata.max() * reg_factor * sparse.eye_array(*ata.shape, format='csc')
        c = self.compute_coordinates(ata)  # coordinate matrix
        self._adjoint_basis = projection_basis @ c  # \hat{b}_i previously known as xi
        return

    def compute_coordinates(self, a: sparse.csc_array) -> sparse.csr_array:
        """Computes coordinate matrix for transformation to reconstruction plane."""
        b = np.eye(a.shape[0])
        c = self.solver.solve(a, b)
        return sparse.csr_array(c)

    def __call__(
        self, 
        data: np.ndarray,
        gmat: np.ndarray | sparse.csr_array | None = None,
        basis: np.ndarray | sparse.sparray | None = None,
    ) -> np.ndarray:
        """Executes the inversion using 
        
        Checks whether decomposition is available and if not performs decomposition before projection.projects images

        Parameters
        ----------
        data : numpy.ndarray
            contains signals or flattened images with shape (#channels, ) or (# channels, # time slices),
            each column of the input represents one time slice
        gmat : scipy.sparse.csr_array, optional
            geometry matrix, required if decomposition was not calculated or provided in init,
            by default None, using previously calculated decomposition stored in the class instance
        kw : dict
            keyword parameters passed to decompose method, used only if gmat is provided and decomposition needs to be calculated, otherwise ignored

        Returns
        -------
        numpy.ndarray
            inversion results with shape (# nodes, # time slices)

        See Also
        --------
        decompose : method handling decomposition of geometry matrix to see supported keyword parameters
        """
        if self._adjoint_basis is None:
            if gmat is None or basis is None:
                msg = 'Decomposition is not calculated. Execute decomposition before inversion or'
                msg += 'provide `gmat` and `basis` to perform decomposition.'
                raise ValueError(msg)
            else:
                basis = sparse.csc_array(basis)
                gmat = sparse.csr_array(gmat)
                self.decompose(gmat, basis)
        coeffs = self._adjoint_basis.T @ data  # coordinates in decomposition basis
        res = self.basis_inv.T @ coeffs
        return res

    def save_decomposition(self, floc: str | Path, description: str = '') -> None:
        """
        Saves decomposition matrix and basis to hdf file. Norms are also included if calculated.

        Parameters
        ----------
        floc : str or pathlib.Path
            file location with name
        description : str, optional
            short user description for file identification
        """
        if self._adjoint_basis is None:
            raise ValueError('Can not save decomposition before it is calculated.')
        floc = str(floc)
        with h5py.File(floc, 'w') as f:
            f.attrs['version'] = '0.2'
            f.attrs['description'] = description
            decomposed_matrix = f.create_group('decomposed_matrix')
            sparse_to_hdf(self._adjoint_basis, decomposed_matrix)
            basis = f.create_group('basis')
            sparse_to_hdf(self._basis, basis)
            if self._norms is not None:
                f.create_dataset('norms', data=self._norms)

    def load_decomposition(self, floc: str | Path) -> None:
        """
        Loads decomposed matrix and basis from an HDF file.

        Norms are loaded only if available in the file.

        Parameters
        ----------
        floc : str or pathlib.Path
            location of hdf file with saved decomposition
        """
        with h5py.File(floc, 'r') as f:
            self._adjoint_basis = hdf_to_sparse(f['decomposed_matrix'])
            self._basis = hdf_to_sparse(f['basis'])
            try:
                self._norms = f['norms'][:]
            except KeyError:
                self._norms = None
        self.basis_inv = sp_linalg.inv(self._basis)

    def normalise(self, precision: float = 1e-6) -> None:
        """
        Computes normalisation factors for decomposition matrix.

        Parameters
        ----------
        precision : float, optional
            neglects decomposition matrix rows with lower norm, by default 1e-6
        """
        adjoint_basis = self._adjoint_basis
        kappa = sparse.linalg.norm(adjoint_basis, axis=0)
        idx = kappa > precision
        norms = np.zeros(kappa.size)
        norms[idx] = (1 / kappa[idx])
        self._norms = norms  # change to expected shape

    def thresholding(self, image: np.ndarray, c: int, precision: float = 1e-6, conv: float = 1e-9) -> np.ndarray:
        """
        Applies thresholding method to provided image.

        Parameters
        ----------
        image : numpy.ndarray
            flattened image, shape (#pixels,)
        c : int
            thresholding sensitivity constant
        precision : float, optional
            normalisation precision, by default 1e-6
        conv : float, optional
            thresholding convergence limit

        Returns
        -------
        numpy.ndarray
            inversion result with threshold applied, shape (#pixels,)

        Raises
        ------
        RuntimeError
            If thresholding is called before decomposition of geometry matrix
        """
        if self._adjoint_basis is None:
            raise RuntimeError('Decomposition must be computed prior to thresholding.')
        if self._norms is None:
            self.normalise(precision)

        if not image.size == self._adjoint_basis.shape[0]:
            raise ValueError('Image size does not match decomposition matrix shape.')
        elif image.ndim > 1:
            image = image.flatten()  # ensure image is flattened to expected shape (#pixels,)
            warnings.warn(
                'Image has more than one dimension, flattening to expected shape (#pixels,).',
                RuntimeWarning,
                stacklevel=2
            )
        # calculate plane basis coefficients
        coeffs = self._adjoint_basis.T @ image

        # thresholding loop
        a = np.abs(coeffs * self._norms)  # normalised coefficients
        threshold_2 = np.sqrt(c**2 / a.size * a.T @ a)
        threshold_1 = 0
        while np.abs(threshold_1-threshold_2) >= conv:
            threshold_1 = threshold_2
            a_temp = a[a <= threshold_1]
            threshold_2 = np.sqrt(c**2 / a_temp.size * a_temp.T @ a_temp)

        # remove plane basis with contributions below threshold and transform to nodes
        coeffs[a < threshold_2] = 0
        out = self._basis @ coeffs
        return out

    def invert(self, data: np.ndarray) -> np.ndarray:
        """
        Uses decomposed matrix to project data into reconstruction plane and then transform to node basis.

        Parameters
        ----------
        data : numpy.ndarray
            contains signals with shape (# channels, # time slices)

        Returns
        -------
        numpy.ndarray
            inversion results with shape (# nodes, # time slices)
        """
        coeffs = self._adjoint_basis.T @ data
        res = coeffs.T @ self.basis_inv
        return res


class SparseInvSolver(Solver):
    """Engine for solving linear systems in BOB decomposition using sparse inverse from scipy."""
    def solve(self, a: np.ndarray | sparse.sparray, b: np.ndarray | sparse.sparray) -> sparse.sparray:
        if isinstance(a, np.ndarray):
            a = sparse.csc_array(a)
        if sparse.issparse(b):
            b = b.toarray()
        if not np.allclose(b, np.eye(a.shape[0])):  # check whether RHS is identity matrix
            raise ValueError('SparseInvSolver is designed to solve for identity matrix as RHS `b`')
        try:
            c = sp_linalg.inv(a)
        except RuntimeError:
            raise ValueError('Singular symmetrized matrix factor. Try increasing regularisation factor.')
        return c
