# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
r"""
Structure of classes is based on algorithms proposed by T. Odstrcil however without sparse optimization

.. [optimized] T. Odstrcil et al., "Optimized tomography methods for plasma emissivity reconstruction at the ASDEX Upgrade tokamak," Rev. Sci. Instrum., 87(12), 123505.
"""
import warnings

import numpy as np
import scipy.sparse as sparse
from scipy.stats.mstats import mquantiles
from scipy.sparse.linalg import eigsh

from .base import RegularisedSolver, RegularisationSelector


class Algebraic(RegularisedSolver):
    """A base class for solvers based on algebraic inversion methods.

    Unlike RegularisedSolver, this class does not support inversion engines, but implements the inversions itself.
    The inversion is performed using series expansion formula utilizing matrix decomposition.
    
    The decomposition is performed in the `decompose` method, which should be implemented in derived classes.

    Attributes
    ----------
    u : numpy.ndarray
        decomposition matrix with shape (#channels, #channels)
    s : numpy.ndarray
        diagonal from a decomposition matrix S with shape (#channels, )
    v : numpy.ndarray
        decomposition matrix with shape (#nodes, #channels)    
    """
    def __init__(self, regularisation_selector=None, num: int | None = None):
        super().__init__(regularisation_selector=regularisation_selector)
        self._engine = None
        self.u: np.ndarray = None
        self.s: np.ndarray = None
        self.v: np.ndarray = None
        self._num = num

    def _cache_inputs(self, data, gmat, regularisation):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.decompose(gmat, regularisation)
        for warning in w:
            warnings.warn(
                f"Warning during decomposition: {warning.message}",
                warning.category,
                stacklevel=2,
            )
        super()._cache_inputs(data, gmat, regularisation)

    def decompose(self, gmat, regularisation, *args, **kwargs):
        """Prepares matrices used in the series expansion.

        This method should be implemented in derived class.
        The matrices are stored in the class attributes `u`, `s`, and `v`.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def invert(self, alpha, data=None, num=None) -> np.ndarray:
        r"""Computes emissivity :math:`g` using provided regularisation parameter `alpha`.
        
        The inversion uses decomposed vectors and series expansion to solve the regularised problem.
        The series expansion is based on the following formula:

        .. math::
            \mathbf{g}(\alpha) = \sum_{i=1}^{m} \frac{k_{i} (\alpha)}{S_{ii}}
            \left( \mathbf{U}^T \cdot \mathbf{f} \cdot \tilde{\mathbf{V}} \right) {}_{*i},

        where :math:`k_i(\alpha)` are so called filtering factors computed using following formula

        .. math::
            k_{i}(\alpha) = \left(1 + \frac{\alpha}{S_{ii}^2} \right)^{-1}


        Parameters
        ----------
        alpha : float
            regularisation parameter
        data : numpy.ndarray, optional
            allows to provide data for inversion directly to this method
            by default None, which means that data provided to the __call__ method will be used
        num : int, optional
            number of columns used for series expansion
            by default the number specified in the class initialization is used

        Returns
        -------
        numpy.ndarray
            results of inversion
        """
        num = num or self._num
        data = data or self._data
        try:
            data = data.reshape(self._gmat.shape[0])
        except ValueError:
            raise ValueError(f"Data shape {data.shape} is not compatible with geometry matrix shape {self._gmat.shape}")
        s = self.s.reshape(1, -1)  # create row vector from the diagonal of the diagonal matrix
        s_sq = np.square(s)
        filters = 1 / (1 + alpha**2 / s_sq)  # alpha**2 to better match Tikhonov scheme
        tmp = filters / s * (self.u.T @ data) * self.v
        g = tmp[:, :num].sum(axis=1)
        return g


class SvdAlgebraic(Algebraic):
    r"""Implements decomposition method using singular value decomposition (SVD) of dense matrices.
    
    References
    ----------
    .. [1] T. Odstrcil et al., "Optimized tomography methods for plasma emissivity reconstruction at the ASDEX Upgrade tokamak," Rev. Sci. Instrum., 87(12), 123505.
    """
    def decompose(
        self,
        gmat: np.ndarray | sparse.spmatrix | sparse.sparray,
        regularisation: np.ndarray | sparse.spmatrix | sparse.sparray
    ):
        if isinstance(gmat, (sparse.spmatrix, sparse.sparray)):
            gmat = gmat.toarray()
            warnings.warn(
                'Converting geometry matrix to dense format for SVD decomposition.', 
                RuntimeWarning,
                stacklevel=2
            )
        if isinstance(regularisation, (sparse.spmatrix, sparse.sparray)):
            regularisation = regularisation.toarray()
            warnings.warn(
                'Converting regularisation matrix to dense format for SVD decomposition.', 
                RuntimeWarning, 
                stacklevel=2
            )
        l_mat = np.linalg.cholesky(regularisation)
        p = np.identity(l_mat.shape[0])
        l_inv = np.linalg.inv(l_mat)
        a = (l_inv @ p) @ gmat.T
        self.u, self.s, vt = np.linalg.svd(a.T, full_matrices=False)
        v = vt.T
        v_tild = (p.T @ l_inv.T) @ v
        self.v = v_tild


class GevAlgebraic(Algebraic):
    r"""Implements decomposition method using generalised eigenvalue decomposition (GEV) of sparse matrices.

    References
    ----------
    .. [1] L.C. Ingesson, "The Mathematics of Some Tomography Algorithms Used at JET," JET Joint Undertaking, 2000
    """
    def decompose(self, gmat: sparse.spmatrix | sparse.sparray, regularisation: sparse.spmatrix | sparse.sparray):
        """Decomposes geometry and regularisation matrices to form suitable for series expansion.

        Parameters
        ----------
        gmat : numpy.ndarray
            geometry matrix with shape (#channels, #nodes), should not be normalised for this method
        regularisation : numpy.ndarray
            regularisation matrix with shape (#nodes, #nodes)
        eigenvalues : int, optional
            number of eigenvalues to be computed, by default None, which means number of nodes eigenvalues will be computed
        """
        if isinstance(gmat, np.ndarray):
            gmat = sparse.csr_array(gmat)
            warnings.warn(
                'Converting geometry matrix to sparse format for GEV decomposition.',
                RuntimeWarning,
                stacklevel=2
            )
        eigenvalues = self._num or gmat.shape[0]
        gdg = gmat.T @ gmat
        s, ev = eigsh(gdg, k=eigenvalues, M=regularisation)
        # flip to have eigenvalues and vectors sorted from largest to smallest
        s = s[::-1]
        ev = ev[..., ::-1]

        s_sqrt = np.sqrt(s)
        self.u = (gmat @ ev) / s_sqrt
        self.s = s
        self.v = s_sqrt * ev


# class QrAlgebraic(Algebraic):
#     def decompose(self, gmat, regularisation):
#         l_mat = np.linalg.cholesky(regularisation)
#         p = np.identity(l_mat.shape[0])
#         l_inv = np.linalg.inv(l_mat)
#         a = l_inv.dot(p.dot(gmat.T))
#         q1, d_roof, s = np.linalg.qr(a.dot(p))
#         q2, r2 = np.linalg.qr(p.dot(s.T))
#         m = d_roof.dot(r2.T).dot(np.linalg.inv(d_roof))
#         r3, d3, q3 = np.linalg.qr(m)


class FastSelector(RegularisationSelector):
    """Fast regularisation parameter selector based on decomposition of a linear algebraic method.

    The regularisation parameter is estimated from the values of the diagonal matrix from the decomposition.
    """
    VALID_METHODS: tuple[str, ...] = ('mean', 'half', 'median', 'quantile', 'logmean')

    def __init__(self, method: str = 'quantile'):
        self.method: str = method

    @property
    def method(self) -> str:
        """The method for fast regularisation parameter selection."""
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        if value not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid method '{value}'. Valid methods are: {', '.join(self.VALID_METHODS)}"
            )
        self._method = value

    def determine(self, solver: Algebraic, method: str | None = None) -> tuple[float, dict[str, float | str]]:
        """Finds regularisation parameter using linear estimate based on values of decomposition diagonal.

        Parameters
        ----------
        solver : Algebraic
            Solver instance containing decomposition spectrum in ``solver.s``.
        method : str {'mean', 'half', 'median', 'quantile', 'logmean'}, optional
            Method for selecting regularisation parameter. If omitted, the selector default is used.
        """
        if not isinstance(solver, Algebraic):
            raise TypeError(
                f"FastSelector can only be used with an Algebraic solver instance, got {type(solver).__name__}."
            )

        method = method or self._method
        if method is None:
            method = 'quantile'
        if solver.s is None:
            raise ValueError('Decomposition spectrum is not available. Run decomposition before determining regularisation.')

        if method == 'mean':
            alpha = solver.s.mean()
        elif method == 'half':
            alpha = solver.s.max() / 2
        elif method == 'median':
            alpha = mquantiles(solver.s, prob=0.5, alphap=0, betap=1)[0]
        elif method == 'quantile':
            quant = 2 / np.e
            alpha = mquantiles(solver.s, prob=quant, alphap=0, betap=1)[0]
        elif method == 'logmean':
            log_s = np.log10(solver.s)
            alpha = np.power(10, log_s.mean())
        else:
            raise ValueError('Unrecognized option for regularisation parameter estimation: {}'.format(method))

        stats = dict(
            method=method,
            alpha=alpha,
            logalpha=np.log10(alpha),
        )
        return alpha, stats
