# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
"""
Contains functions creating regularisation matrices for regularised inversion methods.

 - Derivative matrices on a regular grid, both isotropic and anisotropic based on magnetic flux surfaces (`derivatives`)
 - Combination of derivative matrices into a regularisation matrix (`matrices`)
"""
from .derivatives import all_direction_derivative_matrices, standard_anisotropic_derivative_matrices, derivative_matrix, anisotropic_derivative_matrix, laplace_matrix
from .matrices import weighted_squares


__all__ = [
    "all_direction_derivative_matrices",
    "anisotropic_derivative_matrix",
    "derivative_matrix",
    "laplace_matrix",
    "standard_anisotropic_derivative_matrices",
    "weighted_squares",
]
