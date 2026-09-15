# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
"""
Contains functions and classes required for geometry matrix computation using single line of sight approximation.

 - Functions computing contributions of nodes to gmat
 - reconstruction grid definition
 - line of sight generators for artificial diagnostics
 - gmat handler prototype
"""
from .generators import dense_line, sparse_line, calcam_sparse_line
from .grids import Grid, RegularGrid
from .io import save_sparse_gmat, load_sparse_gmat, save_dense_gmat, load_dense_gmat
from .sightlines import generate_sightlines


__all__ = [
    "Grid",
    "RegularGrid",
    "calcam_sparse_line",
    "dense_line",
    "generate_sightlines",
    "load_dense_gmat",
    "load_sparse_gmat",
    "save_dense_gmat",
    "save_sparse_gmat",
    "sparse_line",
]
