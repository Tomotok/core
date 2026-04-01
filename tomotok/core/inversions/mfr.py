# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
import warnings
import numpy as np
from scipy import sparse

from tomotok.core.regularisations import weighted_squares

from .base import RegularisedSolver


class MinimumFisherRegularisation:
    """
    Implements the Minimum Fisher Regularisation (MFR) method for tomographic inversions.

    Uses an inner loop of regularised inversions to iteratively update the regularisation matrix based on the solution from the previous iteration, effectively allowing for a spatially varying regularisation that adapts to the solution.
    
    Attributes
    ----------
    solver : RegularisedSolver or list of RegularisedSolver
        the solver(s) used in the inner loop of Minimum Fisher Regularisation
        if a single solver is provided, it will be used for all MFI iterations, otherwise the number of solvers should match the number of MFI iterations
    """
    def __init__(self, solver: RegularisedSolver | list[RegularisedSolver]):
        if isinstance(solver, list):
            if not all(isinstance(s, RegularisedSolver) for s in solver):
                msg = 'All elements of the solver list should be instances of RegularisedSolver.'
                raise ValueError(msg)
            self.solver = solver
        elif isinstance(solver, RegularisedSolver):
            self.solver = solver
        else:
            msg = 'Solver should be either an instance of RegularisedSolver or a list of RegularisedSolver instances.'
            raise ValueError(msg)
        self.solver = solver
    
    def __call__(
        self, 
        data: np.ndarray,
        gmat: np.ndarray | sparse.csc_array | sparse.csr_array, 
        derivatives: list[sparse.csc_array | sparse.csr_array],
        errors: float | np.ndarray,
        derivative_weights: list[float] | float | None = None,
        mfi_num: int = 3,
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
        mfi_num : int, optional
            number of minimum Fisher loops, by default 3
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

        if isinstance(self.solver, list):
            if len(self.solver) != mfi_num:
                msg = 'Length of the solver list should be equal to the number of MFI loops.'
                msg += f' Got {len(self.solver)} solvers and mfi_num={mfi_num}.'
                raise ValueError(msg)
        elif isinstance(self.solver, RegularisedSolver):
            self.solver = [self.solver] * mfi_num

        # MFI loop (outer)
        for solver in self.solver:
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
                inversion, stats = solver(data, gmat, regularisation, errors)
            for warning in w:
                warnings.warn(
                    f"MFR solver: {warning.message}",
                    warning.category,
                    stacklevel=2,
                )
            statistics.append(stats)
            g = inversion
        return g, statistics
