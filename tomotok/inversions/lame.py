# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
r"""
Structure of classes is based on algorithms proposed by T. Odstrcil [optimized]_.

References
----------
.. [optimized] T. Odstrcil et al., "Optimized tomography methods for plasma emissivity reconstruction at the ASDEX Upgrade tokamak," Rev. Sci. Instrum., 87(12), 123505.
"""
import warnings

import numpy as np
from scipy import sparse
from scipy.stats.mstats import mquantiles
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, solve_triangular

from .base import RegularisedInversion, RegularisationSelector


class Algebraic(RegularisedInversion):
    """A base class for solvers based on algebraic inversion methods.

    Unlike RegularisedInversion, this class does not support inversion solvers, but implements the inversions itself.
    The inversion is performed using series expansion formula utilizing matrix decomposition.
    
    The decomposition is performed in the `decompose` method, which should be implemented in derived classes.

    Attributes
    ----------
    u : numpy.ndarray
        decomposition matrix with shape (#channels, K), where K is the number of stored components
    s : numpy.ndarray
        (generalised) singular values sorted from largest to smallest,
        i.e. diagonal of the decomposition matrix S with shape (K, )
    v : numpy.ndarray
        decomposition matrix with shape (#nodes, K)
    """
    def __init__(self, regularisation_selector=None, num: int | None = None):
        """
        Parameters
        ----------
        regularisation_selector : RegularisationSelector, optional
            selector for regularisation parameter, by default None, which means that the regularisation parameter should be provided directly to the `invert` method
        num : int, optional
            number of components used for series expansion, by default None, which means that all components are used.
            For GevAlgebraic, it also sets the number of computed eigenpairs, see `GevAlgebraic.decompose`.
        """
        super().__init__(regularisation_selector=regularisation_selector)
        self._solver = None
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
        """Prepare matrices used in the series expansion.

        This method should be implemented in derived class.
        The matrices are stored in the class attributes `u`, `s`, and `v`.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def invert(self, alpha, data=None, num=None) -> np.ndarray:
        r"""Compute emissivity :math:`g` using provided regularisation parameter `alpha`.
        
        The inversion uses decomposed vectors and series expansion to solve the regularised problem.
        The series expansion is based on the following formula:

        .. math::
            \mathbf{g}(\alpha) = \sum_{i=1}^{K} \frac{k_{i} (\alpha)}{S_{ii}}
            \left( \mathbf{u}_{i} \cdot \mathbf{f} \right) \mathbf{v}_{i} \,,

        where :math:`K` is the number of components used (given by `num`, at most the number of stored components),
        :math:`\mathbf{u}_i` and :math:`\mathbf{v}_i` are the i-th columns of `u` and `v`,
        :math:`S_{ii}` are the (generalised) singular values stored in `s`
        and :math:`k_i(\alpha)` are filter factors computed using the following formula

        .. math::
            k_{i}(\alpha) = \left(1 + \frac{\alpha}{S_{ii}^2} \right)^{-1}


        Parameters
        ----------
        alpha : float
            regularisation parameter
        data : numpy.ndarray, optional
            allows to provide data for inversion directly to this method, the data must be divided by the measurement errors.
            By default None, which means that data provided to the __call__ method will be used.
        num : int, optional
            number of components used for series expansion.
            By default None, which means that the number specified in the class initialization is used.

        Returns
        -------
        numpy.ndarray
            results of inversion
        """
        num = num or self._num
        if data is None:
            data = self._data
        try:
            data = data.reshape(self._gmat.shape[0])
        except ValueError:
            raise ValueError(f"Data shape {data.shape} is not compatible with geometry matrix shape {self._gmat.shape}")
        s = self.s.reshape(1, -1)  # create row vector from the diagonal of the diagonal matrix
        s_sq = np.square(s)
        filters = 1 / (1 + alpha / s_sq)
        tmp = filters / s * (self.u.T @ data) * self.v
        g = tmp[:, :num].sum(axis=1)
        return g


class SvdAlgebraic(Algebraic):
    r"""Implements decomposition method using singular value decomposition (SVD) of dense matrices.

    The regularisation matrix must be positive definite, since its Cholesky decomposition is computed.
    A semi-definite matrix composed only of derivative matrices can be made positive definite
    by adding a small multiple of the identity matrix.

    References
    ----------
    .. [1] T. Odstrcil et al., "Optimized tomography methods for plasma emissivity reconstruction at the ASDEX Upgrade tokamak," Rev. Sci. Instrum., 87(12), 123505.
    """
    def decompose(
        self,
        gmat: np.ndarray | sparse.spmatrix | sparse.sparray,
        regularisation: np.ndarray | sparse.spmatrix | sparse.sparray
    ):
        """Decompose geometry and regularisation matrices to form suitable for series expansion.

        All min(#channels, #nodes) components are computed.

        Parameters
        ----------
        gmat : numpy.ndarray or scipy.sparse.sparray
            geometry matrix with shape (#channels, #nodes), sparse input is converted to dense format
        regularisation : numpy.ndarray or scipy.sparse.sparray
            regularisation matrix with shape (#nodes, #nodes), must be symmetric positive definite,
            sparse input is converted to dense format
        """
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
        # triangular solves instead of explicit inversion of the Cholesky factor
        a = solve_triangular(l_mat, gmat.T, lower=True)
        self.u, self.s, vt = np.linalg.svd(a.T, full_matrices=False)
        self.v = solve_triangular(l_mat, vt.T, lower=True, trans='T')


class GevAlgebraic(Algebraic):
    r"""Implements decomposition method using generalised eigenvalue decomposition (GEV).

    Sparse eigenvalue solver is used when only a part of the spectrum is requested,
    which makes the method suitable for large grids.

    The regularisation matrix must be positive definite. A singular regularisation matrix is not detected
    and leads to an incorrect result. A semi-definite matrix composed only of derivative matrices
    can be made positive definite by adding a small multiple of the identity matrix.

    References
    ----------
    .. [1] L.C. Ingesson, "The Mathematics of Some Tomography Algorithms Used at JET," JET Joint Undertaking, 2000
    """
    def decompose(
        self,
        gmat: np.ndarray | sparse.spmatrix | sparse.sparray,
        regularisation: np.ndarray | sparse.spmatrix | sparse.sparray
    ):
        """Decompose geometry and regularisation matrices to form suitable for series expansion.

        The number of computed eigenvalues is given by the ``num`` parameter, by default the number of channels.
        If it is not smaller than the number of nodes, the sparse eigenvalue solver cannot be used
        and the full dense generalised eigenvalue problem is solved instead.
        Components with eigenvalues that are numerically zero, e.g. for rank-deficient geometry matrix,
        do not contribute to the solution and are discarded, so fewer components than requested may be stored.

        Parameters
        ----------
        gmat : scipy.sparse.sparray or numpy.ndarray
            geometry matrix with shape (#channels, #nodes), dense input is converted to sparse format
        regularisation : scipy.sparse.sparray or numpy.ndarray
            regularisation matrix with shape (#nodes, #nodes), must be symmetric positive definite
        """
        if isinstance(gmat, np.ndarray):
            gmat = sparse.csr_array(gmat)
            warnings.warn(
                'Converting geometry matrix to sparse format for GEV decomposition.',
                RuntimeWarning,
                stacklevel=2
            )   
        nodes = gmat.shape[1]
        eigenvalues = self._num or gmat.shape[0]
        gdg = gmat.T @ gmat
        if eigenvalues < nodes:
            s, ev = eigsh(gdg, k=eigenvalues, M=regularisation)
        else:
            if isinstance(regularisation, (sparse.spmatrix, sparse.sparray)):
                regularisation = regularisation.toarray()
            s, ev = eigh(gdg.toarray(), regularisation)
            s = s[-eigenvalues:]
            ev = ev[:, -eigenvalues:]
        # flip to have eigenvalues and vectors sorted from largest to smallest
        s = s[::-1]
        ev = ev[..., ::-1]

        # discard numerically zero (or negative due to round-off) eigenvalues
        tol = s[0] * max(gdg.shape) * np.finfo(s.dtype).eps
        keep = s > tol
        if not keep.all():
            warnings.warn(
                f'Discarding {np.count_nonzero(~keep)} components with numerically zero eigenvalues.',
                RuntimeWarning,
                stacklevel=2
            )
            s = s[keep]
            ev = ev[:, keep]

        # eigenvalues are squares of generalised singular values, which are expected by the series expansion
        s_sqrt = np.sqrt(s)
        self.u = (gmat @ ev) / s_sqrt
        self.s = s_sqrt
        self.v = ev


class FastSelector(RegularisationSelector):
    """Fast regularisation parameter selector based on decomposition of a linear algebraic method.

    The regularisation parameter is estimated from the values of the diagonal matrix from the decomposition,
    i.e. from the (generalised) singular values. By default, the estimate is squared, since the singular values
    enter the filter factors squared and the regularisation parameter has the scale of their squares.

    Notes
    -----
    This selector is designed exclusively for ``Algebraic`` solver instances (``SvdAlgebraic``, ``GevAlgebraic``)
    and will raise a ``TypeError`` if used with other solver types such as ``Tikhonov``.
    """
    VALID_METHODS: tuple[str, ...] = ('mean', 'half', 'median', 'quantile', 'logmean')

    def __init__(self, method: str = 'quantile', square: bool = True):
        """
        Parameters
        ----------
        method : str {'mean', 'half', 'median', 'quantile', 'logmean'}, optional
            statistic of the singular values used as the estimate, by default 'quantile'.
            The options are the arithmetic mean, half of the maximum, the median, the 2/e quantile
            and the geometric mean (logmean). The statistic is computed before squaring,
            e.g. 'half' yields max(s)**2 / 4 when `square` is True.
        square : bool, optional
            whether the estimate is squared to match the scale of the regularisation parameter, by default True.
            If False, the estimate is returned on the scale of the singular values.
        """
        self.method: str = method
        self.square: bool = square

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
        """Find regularisation parameter using linear estimate based on values of decomposition diagonal.

        Parameters
        ----------
        solver : Algebraic
            Solver instance containing decomposition spectrum in ``solver.s``.
        method : str {'mean', 'half', 'median', 'quantile', 'logmean'}, optional
            Method for selecting regularisation parameter. If omitted, the selector default is used.

        Returns
        -------
        alpha : float
            estimated regularisation parameter
        stats : dict
            used method, square flag, estimated regularisation parameter and its decimal logarithm
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
            raise ValueError(f'Unrecognized option for regularisation parameter estimation: {method}')
        if self.square:
            alpha = alpha**2

        stats = {
            "method": method,
            "square": self.square,
            "alpha": alpha,
            "logalpha": np.log10(alpha),
        }
        return alpha, stats
