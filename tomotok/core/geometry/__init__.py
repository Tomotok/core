# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Contains functions and classes required for geometry matrix computation using single line of sight approximation.

 - Functions computing contributions of nodes to gmat
 - reconstruction grid definition
 - line of sight generators for artificial diagnostics
 - gmat handler prototype
"""
from .generators import sparse_line_3d, calcam_sparse_line_3d
from .grids import RegularGrid
from .los import generate_los, generate_directions
from .io import save_sparse_gmat, load_sparse_gmat
