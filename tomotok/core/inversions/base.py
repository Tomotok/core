from typing import Tuple

import numpy as np


class Solver(object):
    """
    Base class for inversion problem solvers.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, data, gmat, *args, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Solves the inversion problem. 

        Parameters
        ----------
        data : np.ndarray
            The data to be inverted.
        gmat : np.ndarray
            The forward model matrix.
        *args : tuple
            Additional arguments for the solver.
            Typically, this includes the regularisation matrix and estimated errors.
        **kwargs : dict
            Additional keyword arguments for the solver.

        Returns
        -------
        np.ndarray
            The inversion result
        dict
            A dictionary containing statistics about the inversion process
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class RegularisedSolver(Solver):
    """
    Base class for solvers that use regularisation.
    """

    def __call__(self, data, gmat, regularisation, errors, **kwargs):
        """
        Solves the inversion problem with regularisation.

        Parameters
        ----------
        data : np.ndarray
            The data to be inverted.
        gmat : np.ndarray
            The forward model matrix.
        regularisation : np.ndarray
            The regularisation matrix.
        errors : np.ndarray
            The estimated errors.
        kwargs
            Additional keyword arguments for the solver.

        Returns
        -------
        np.ndarray
            The inversion result
        dict
            A dictionary containing statistics about the inversion process
        """
        return super().__call__(data, gmat, regularisation, errors, **kwargs)
