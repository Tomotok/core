# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Contains inversion class for Biorthogonal Basis Decomposition Algorithm using single node basis [1]_ derived from wavelet-vaguelette decomposition [2]_

.. [1] Jordan Cavalier et al 2019 Nucl. Fusion 59 056025
.. [2] R. Nguyen van yen et al 2011 Nucl. Fusion 52 013005
"""
import warnings

import numpy as np
import scipy.sparse as sparse


class Bob(object):
    """
    BiOrthogonal Basis decomposition

    Parameters
    ----------
    dec_mat : scipy.sparse.csr_matrix, optional
        previsously decomposed matrix, avoids recomputration of decomposition when provided
    basis : array-like, optional
        tbd
    precision : float, optional
        precision used for normalisation of decomposition matrix, by default 1e-6 

    Attributes
    ----------
    basis : array-like
        :math:`\mathbf{b}_i` basis vectors of reconstruction plane
    dec_mat : scipy.sparse.csr_matrix
        :math:`\hat{\mathbf{e}}_i` decomposed matrix used to transform image into reconstruction plane
    dec_mat_normed : scipy.sparse.csr_matrix
        normalised decomposed matrix
    precision : float
        precision used for normalisation of decomposition matrix
    nnodes : int
        number of nodes in reconstruction plane
    """

    def __init__(self, dec_mat=None, basis=None, precision=1e-6):
        super().__init__()
        self.basis = basis
        self.precision = precision
        self.dec_mat = dec_mat
        self.dec_mat_normed = None
        self.nnodes = None
        return

    def create_basis(self, *args, **kwargs):
        raise NotImplementedError()

    # TODO add basis as a parameter
    def decompose(self, gmat):
        """
        Decomposes the geometry matrix using basis vectors

        Parameters
        ----------
        gmat : sparse.csr_matrix
            geometry matrix
        """
        if gmat.shape[0] < gmat.shape[1]:
            warnings.warn('Biorthogonal algorithm requires more lines of sights than nodes in reconstruction plane to run reliably')
        self.nnodes = gmat.shape[1]
        if self.basis is None:
            self.create_basis()
        chi = gmat.dot(self.basis)
        a = chi.T.dot(chi).toarray()
        b = np.eye(self.nnodes)
        res = np.linalg.lstsq(a, b)
        c = sparse.csr_matrix(res[0])
        xi = chi.dot(c)
        self.dec_mat = xi
        return

    def __call__(self, data, gmat=None, **kw):
        """
        Decomposes geometry matrix and projects images

        Parameters
        ----------
        data : np.ndarray
            contains signals with shape ('#chnl', '#timeslices')
        gmat : scipy.csr_matrix
            geometry matrix
        grid : tomotok.io.Pixgrid
            reconstruction grid
        """
        if self.dec_mat is None:
            if gmat is None:
                raise ValueError('Gmat must be provided for decomposition')
            else:
                self.decompose(gmat)
        res = self.dec_mat.T.dot(data)
        return res

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
        sparse.save_npz(floc, self.dec_mat, compressed=True)

    def normalise(self, precision=None):
        if precision is None:
            precision = self.precision
        xi = self.dec_mat
        kappa = sparse.linalg.norm(xi, axis=0)
        idx = kappa > self.precision
        norms = np.zeros(kappa.size)
        norms[idx] = (1 / kappa[idx])
        xi_norm = xi.multiply(sparse.csr_matrix(norms))
        self.dec_mat_normed = xi_norm

    def thresholding(self, image, c=4, precision=None):
        """
        Applies thresholding method to provided image.

        Parameters
        ----------
        image : np.ndarray
            flattened image with shape (#pixels, 1)
        c : int, optional
            thresholding sensitivity constant, by default 4
        precision : float, optional
            normalisation precision

        Returns
        -------
        np.ndarray
            thresholded image

        Raises
        ------
        RuntimeError
            If tresholding is called before decomposition of geometry matrix
        """
        if self.dec_mat is None:
            raise RuntimeError('Decompose first, threshold later')
        if self.dec_mat_normed is None:
            self.normalise(precision)
        temp = sparse.csr_matrix(image.T)
        a = np.abs(temp.dot(self.dec_mat_normed).toarray())

        threshold_2 = np.sqrt(c**2 / np.size(a) * a.dot(a.T))
        threshold_1 = 0
        while np.abs(threshold_1-threshold_2) >= 1e-9:
            threshold_1 = threshold_2
            a_temp = a[a <= threshold_1]
            threshold_2 = np.sqrt(c**2 / np.size(a_temp) * a_temp.dot(a_temp.T))

        out = self.dec_mat.T.dot(image)
        out[a.T < threshold_2] = 0
        return out


class SimpleBob(Bob):
    def create_basis(self):
        self.basis = sparse.eye(self.nnodes)
