# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
import warnings
import numpy as np
from scipy import sparse

from tomotok.regularisations import weighted_squares

from .base import RegularisedInversion


class MinimumFisherRegularisation:
    """
    Implements the Minimum Fisher Regularisation (MFR) method for tomographic inversions.

    Uses an inner loop of regularised inversions to iteratively update the regularisation matrix based on the solution from the previous iteration, effectively allowing for a spatially varying regularisation that adapts to the solution.

    Attributes
    ----------
    inversion_list : list of RegularisedInversion
        inversions used in the inner loop, one for each MFI iteration
    termination_method : str
        method used to terminate the MFI loop
    mfi_num_max : int
        maximum number of MFI iterations
    weight_floor : float
        minimum value of the previous solution relative to its maximum used when computing node weights
    """
    TERMINATION_METHODS: tuple[str, ...] = ("max_iter",)

    def __init__(
        self,
        inversion: RegularisedInversion | list[RegularisedInversion],
        termination_method: str = "max_iter",
        mfi_num_max: int = 3,
        weight_floor: float = 1e-6,
    ):
        """
        Parameters
        ----------
        inversion : RegularisedInversion or list of RegularisedInversion
            inversion used in the inner loop, i.e. for determination of the regularisation parameter and solution
            of the regularised problem, usually :class:`~tomotok.inversions.Tikhonov`,
            if a single inversion is provided, it is used for all MFI iterations, otherwise
            each inversion is used for the corresponding MFI iteration and their number must be equal to `mfi_num_max`
        termination_method : str, optional
            method used to terminate the MFI loop, by default "max_iter", which is currently the only supported method
            and stops the loop after `mfi_num_max` iterations
        mfi_num_max : int, optional
            maximum number of MFI iterations, by default 3
        weight_floor : float, optional
            minimum value of the previous solution relative to its maximum used when computing
            node weights (``1 / g``) for the next MFI iteration, by default 1e-6,
            prevents near-zero or negative values (e.g. from a non-negativity-constrained solver)
            from producing unbounded weights and an ill-conditioned regularisation matrix

        Raises
        ------
        TypeError
            if `inversion` is not a RegularisedInversion or a list of them
        ValueError
            if the length of the inversion list differs from `mfi_num_max`
            or if `termination_method` is not supported
        """
        if termination_method not in self.TERMINATION_METHODS:
            msg = f'Unsupported termination method {termination_method!r}.'
            msg += f' Supported methods are: {", ".join(self.TERMINATION_METHODS)}.'
            raise ValueError(msg)
        if isinstance(inversion, list):
            if not all(isinstance(s, RegularisedInversion) for s in inversion):
                msg = 'All elements of the inversion list should be instances of RegularisedInversion.'
                raise ValueError(msg)
            if len(inversion) != mfi_num_max:
                msg = 'Length of the inversion list should be equal to the maximum number of MFI loops.'
                msg += f' Got {len(inversion)} inversions and mfi_num_max={mfi_num_max}.'
                raise ValueError(msg)
            self.inversion_list = inversion
        elif isinstance(inversion, RegularisedInversion):
            self.inversion_list = [inversion] * mfi_num_max
        else:
            msg = 'Inversion should be either an instance of RegularisedInversion or a list of RegularisedInversion instances.'
            raise TypeError(msg)
        self.termination_method = termination_method
        self.mfi_num_max = mfi_num_max
        self.weight_floor = weight_floor

    def __call__(
        self, 
        data: np.ndarray,
        gmat: np.ndarray | sparse.csc_array | sparse.csr_array, 
        derivatives: list[sparse.csc_array | sparse.csr_array],
        errors: float | np.ndarray,
        derivative_weights: list[float] | list[np.ndarray] | None = None,
        initial_guess: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Inversion using the Minimum Fisher Regularisation method.

        Parameters
        ----------
        data : numpy.ndarray
            measured signals with shape (#channels,)
        gmat : numpy.ndarray or scipy.sparse.csr_array or scipy.sparse.csc_array
            geometry matrix with shape (#channels, #nodes)
        derivatives : list of scipy.sparse.csc_array or scipy.sparse.csr_array
            derivative matrices with shape (#nodes, #nodes) used to construct the regularisation matrix
        errors : float or numpy.ndarray
            expected errors of the measured signals, a single value or an array with the shape of `data`,
            used for normalisation of the inputs and in the Pearson test
        derivative_weights : list of float or list of array-like, optional
            anisotropy coefficients, i.e. weights of the individual derivative matrices, normalised to unit sum,
            a list of arrays with shape (#nodes,) specifies the weight of each derivative matrix in each node
            (row of the matrix), the weights are then normalised to unit sum in each node separately,
            by default None, which corresponds to equal weights of all derivative matrices
        initial_guess : numpy.ndarray, optional
            initial estimate of the solution with shape (#nodes,), used to compute node weights in the first MFI iteration,
            by default None, which corresponds to equal node weights, i.e. linear regularisation in the first iteration

        Returns
        -------
        numpy.ndarray
            inversion result with shape (#nodes,)
        list of dict
            statistics returned by the inner inversion in each MFI iteration
        """
        if initial_guess is None:
            g = np.ones(gmat.shape[1])
        else:
            g = initial_guess
        statistics = []

        # MFI loop (outer)
        for inversion in self.inversion_list:
            g_floor = np.maximum(g, g.max() * self.weight_floor)
            node_weights = 1 / g_floor
            regularisation = weighted_squares(
                derivatives,
                derivative_weights,
                node_weights,
            )
            # Regularisation loop (inner)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                g, stats = inversion(data, gmat, regularisation, errors)
            for warning in w:
                warnings.warn(
                    f"MFR solver: {warning.message}",
                    warning.category,
                    stacklevel=2,
                )
            statistics.append(stats)
        return g, statistics
