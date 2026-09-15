# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar

from .solvers import Solver, CholeskySolver


class Inversion:
    """Base class for inversion problems."""
    def __init__(self, solver: Solver | None = None):
        self.solver: Solver | None = solver

    @property
    def solver(self) -> Solver | None:
        """Algebraic backend solving the system of linear equations."""
        return self._solver

    @solver.setter
    def solver(self, value: Solver | None):
        if isinstance(value, Solver) or value is None:
            self._solver = value
        else:
            raise ValueError("Solver must be an instance of the Solver class.")

    def __call__(
        self,
        data: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Executes the entire inversion scheme, including input processing.

        Parameters
        ----------
        data : np.ndarray
            The data to be inverted.
        *args : tuple
            Additional arguments for the solver.
            Typically, this includes the regularisation matrix and estimated errors.
        **kwargs : dict
            Additional keyword arguments for the solver.

        Returns
        -------
        np.ndarray
            The inversion result
        dict[str, Any]
            A dictionary containing statistics about the inversion process
        """
        raise NotImplementedError("This method should be defined by subclasses.")


class RegularisedInversion(Inversion):
    """Base class for inversion methods that use regularisation."""
    def __init__(
        self, 
        solver: Solver | None = None,
        regularisation_selector: RegularisationSelector | None = None,
    ):
        """
        Parameters
        ----------
        solver : Solver or None, optional
            The solver used for solving the inversion problem. If ``None`` (default), a new :class:`CholeskySolver` instance is created and used for this inversion instance.
        regularisation_selector : RegularisationSelector or None, optional
            Strategy object used to determine the regularisation parameter.
            If ``None`` (default), a new :class:`PearsonSelector` instance is created
            and used for this inversion instance.
        """
        solver = solver or CholeskySolver()
        super().__init__(solver=solver)
        self._data: np.ndarray | None = None
        self._gmat: sparse.spmatrix | sparse.sparray | None = None
        self._regularisation: sparse.spmatrix | sparse.sparray | np.ndarray | None = None
        if regularisation_selector is None:
            self.selector = PearsonSelector()
        elif isinstance(regularisation_selector, RegularisationSelector):
            self.selector = regularisation_selector
        else:
            raise TypeError(
                "regularisation_selector must be a RegularisationSelector instance or None"
            )

    def _cache_inputs(
        self,
        normalised_data: np.ndarray,
        normalised_gmat: sparse.spmatrix | sparse.sparray | np.ndarray,
        regularisation: sparse.spmatrix | sparse.sparray | np.ndarray,
    ) -> None:
        self._data = normalised_data
        self._gmat = normalised_gmat
        self._regularisation = regularisation

    def __call__(
        self,
        data: np.ndarray,
        gmat: sparse.sparray | np.ndarray,
        regularisation: sparse.sparray | np.ndarray,
        errors: float | np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Executes the regularised inversion scheme.

        Normalises the data and geometry matrix using the provided error estimates, caches the normalised inputs, determines the regularisation parameter using the provided selector, and finally inverts the problem using the determined regularisation parameter.

        Parameters
        ----------
        data : np.ndarray
            The data to be inverted.
        gmat : sparse.spmatrix or sparse.sparray or np.ndarray
            The geometry matrix.
        regularisation : sparse.spmatrix or sparse.sparray or np.ndarray
            The regularisation matrix.
        errors : float or np.ndarray
            The error estimates for the data that are used for normalisation in chi-squared test.
            If a single float is provided, it is assumed that the error is constant for all data points.

        Returns
        -------
        np.ndarray
            The inversion result.
        dict[str, Any]
            A dictionary containing statistics about the inversion process.
        """
        if isinstance(errors, (int, float)):  # constant error estimate for all channels
            errors = np.full_like(data, errors)
        elif isinstance(errors, np.ndarray) and errors.shape != data.shape:
            raise ValueError("Shape of errors array must match shape of data array.")
        data_nrm = data / errors
        norms = sparse.diags_array(1/errors)
        gmat_nrm = norms @ gmat
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._cache_inputs(data_nrm, gmat_nrm, regularisation)
            alpha, stats = self.determine_regularisation()
            out = self.invert(alpha)
        for warning in w:
            warnings.warn(
                f"Warning during inversion: {warning.message}",
                warning.category,
                stacklevel=2,
            )
        return out, stats

    def invert(self, alpha: float) -> np.ndarray:
        """Inverts the regularised problem using provided regularisation parameter.

        Uses the standard inputs to produce the regularised problem and then uses the solve method to find the solution.

        Parameters
        ----------
        alpha : float
            Regularisation parameter value

        Returns
        -------
        np.ndarray
            The inversion result
        """
        raise NotImplementedError("This method should be defined by subclasses.")

    def determine_regularisation(self, *args: Any, **kwargs: Any) -> tuple[float, dict[str, Any]]:
        """Determines the value of the regularisation parameter to be used in the inversion.

        Parameters
        ----------
        *args : tuple
            Additional arguments for the solver.
        **kwargs : dict
            Additional keyword arguments for the solver.

        Returns
        -------
        float
            The regularisation parameter value.
        dict[str, Any]
            A dictionary containing statistics about the inversion process.
        """
        return self.selector.determine(self)


class RegularisationSelector:
    """Base class for regularisation parameter selectors."""
    def determine(self, inversion: RegularisedInversion) -> tuple[float, dict[str, Any]]:
        raise NotImplementedError("This method should be defined by subclasses.")


class PearsonSelector(RegularisationSelector):
    """Implements regularisation based on Pearson's chi-squared test."""
    def __init__(
        self,
        bounds: tuple[float, float] = (-30, 10),
        iter_max: int = 50,
        tolerance: float = 1e-4,
    ):
        self._bounds = bounds
        self._iter_max = iter_max
        self._tolerance = tolerance
        self._chisq: float = np.nan
        self._best_logalpha: float = np.nan
        self._best_distance: float = float("inf")

    def _pearson_test(self, inversion: RegularisedInversion, g: np.ndarray) -> float:
        r"""
        Computes retrofit and residuum :math:`\chi^2` using pearson test on normalised data and geometry matrix.

        The test is defined as:

        .. math ::
            \chi^2 = \frac{1}{M} \sum_{i}^{M} \left(\tilde{\mathbf{f}} - \tilde{\mathbf{T}} \cdot \mathbf{g} \right)_i^2

        Where the tilde denotes normalisation by measurement errors.

        Parameters
        ----------
        inversion : RegularisedInversion
            The inversion for which the test is performed. The test uses the data and geometry matrix stored in the inversion instance.
        g : numpy.ndarray
            vector of tested emissivity

        Returns
        -------
        float
        """
        retrofit = inversion._gmat @ g
        misfit = retrofit - inversion._data
        misfit_sq = np.power(misfit, 2)
        chisq = np.average(misfit_sq)
        return chisq

    def _test_regularization(self, logalpha: float, inversion: RegularisedInversion) -> float:
        """
        Function passed to minimisation routine used for finding regularisation parameter value.

        Inverses signals using given regularisation parameter and computes chi2 test

        Stores pearson test result in attribute last_chi.

        Parameters
        ----------
        logalpha : float
            natural logarithm of regularisation parameter

        Returns
        -------
        abs(chi2 - 1) : float
            1D Euclidean distance from ideal Pearson test result
        """
        alpha = 10**logalpha
        g = inversion.invert(alpha)
        chi2 = self._pearson_test(inversion, g)
        distance = abs(chi2 - 1)
        if distance < self._best_distance:
            self._chisq = chi2
            self._best_distance = distance
            self._best_logalpha = logalpha
        return distance

    def determine(self, inversion: RegularisedInversion) -> tuple[float, dict[str, Any]]:
        """
        Determines value of regularisation parameter using minimisation of Pearson test.
        
        The minimisation is done using the `minimize_scalar` function from `scipy.optimize`.

        Parameters
        ----------
        bounds : tuple
            The bounds for the regularisation parameter.
        iter_max : int
            The maximum number of iterations.
        tolerance : float
            The tolerance for convergence.

        Returns
        -------
        sparse.spmatrix
            The regularisation matrix.
        dict
            A dictionary containing statistics about the inversion process.
        """
        self._chisq = np.nan
        self._best_logalpha = np.nan
        self._best_distance = float("inf")
        res = minimize_scalar(
            self._test_regularization,
            args=(inversion,),
            method='bounded',
            bounds=self._bounds,
            options={'maxiter': self._iter_max, 'xatol': self._tolerance},
        )
        if res.status == 1:
            warnings.warn(
                'Maximum number of iteration in regularisation parameter search. Consider increasing iter_max.',
                RuntimeWarning,
                stacklevel=3,
            )
        alpha = 10**res.x
        chisq = self._chisq if self._chisq is not None else np.nan
        stats = {
            "iter_num": res.nfev,
            "logalpha": res.x,
            "alpha": alpha,
            "chisq": chisq
        }
        return alpha, stats


class FixedSelector(RegularisationSelector):
    """Provides fixed value of regularisation parameter."""
    def __init__(self, value: float):
        """
        Parameters
        ----------
        value : float
            The fixed value of the regularisation parameter.
        """
        self._value = value

    def determine(self, inversion: RegularisedInversion | None = None) -> tuple[float, dict[str, Any]]:
        """Determines value of regularisation parameter using the fixed value.

        Returns
        -------
        alpha : float
            Regularisation parameter value.
        stats : dict
            A dictionary containing statistics about the inversion process.
        """
        alpha = self._value
        # TODO: decide on what statistics to return here. Chi2?
        stats = {
            "iter_num": 0,
            "logalpha": np.log10(alpha),
            "alpha": alpha,
        }
        return alpha, stats
