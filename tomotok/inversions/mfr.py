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
    inversion : RegularisedInversion or list of RegularisedInversion
        the inversion(s) used in the inner loop of Minimum Fisher Regularisation
        if a single inversion is provided, it will be used for all MFI iterations, otherwise 
        each inversion is used for the corresponding MFI iteration, the number of inversions must match mfi_num_max
    termination_method : str
        the method used to terminate the MFI loop, currently only "max_iter" is supported
    mfi_num_max : int
        the maximum number of MFI loops before the iteration is stopped
    """
    def __init__(
        self, 
        inversion: RegularisedInversion | list[RegularisedInversion],
        termination_method: str = "max_iter",
        mfi_num_max: int = 3,
    ):
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

    def __call__(
        self, 
        data: np.ndarray,
        gmat: np.ndarray | sparse.csc_array | sparse.csr_array, 
        derivatives: list[sparse.csc_array | sparse.csr_array],
        errors: float | np.ndarray,
        derivative_weights: list[float] | float | None = None,
        initial_guess: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Inversion using the Minimum Fisher Regularisation method.

        Parameters
        ----------
        data : np.ndarray
        gmat : scipy.sparse.spmatrix
        derivatives : list of scipy.sparse.spmatrix
            list of derivative matrices with shape (#nodes, #nodes)
        errors : np.ndarray
            error estimates for the data, used in the weighting of the data misfit term in the inversion
        derivative_weights : list of floats or list of array-like, optional
            anisotropy of derivatives used in the constrution of regularisation matrix,
            by default None, which corresponds to isotropic regularisation (all weights equal to 1)
        mfi_num_max : int, optional
            number of minimum Fisher loops before the iteration is stopped, by default 3
        initial_guess : np.ndarray, optional
            initial guess for the solution, used in the first MFI loop to compute node weights for regularisation matrix,
            by default None, which corresponds to a uniform initial guess of 1

        Returns
        -------
        np.ndarray
            inversion result
        list
            list of inversion statistics for each inner loop of the MFR algorithm
        """
        if initial_guess is None:
            g = np.ones(gmat.shape[1])
        else:
            g = initial_guess
        statistics = []

        # MFI loop (outer)
        for inversion in self.inversion_list:
            node_weights = 1 / g
            node_weights[g <= 0] = np.nanmax(node_weights)
            regularisation = weighted_squares(
                derivatives,
                derivative_weights,
                node_weights,
            )
            # Regularisation loop (inner)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                inversion, stats = inversion(data, gmat, regularisation, errors)
            for warning in w:
                warnings.warn(
                    f"MFR solver: {warning.message}",
                    warning.category,
                    stacklevel=2,
                )
            statistics.append(stats)
            g = inversion
        return g, statistics
