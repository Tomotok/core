# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
"""
The heart of the package containing the inversion algorithms.

The basis is formed by solver classes that are responsible for the actual inversion of the problem with or without regularisation.
The most general is `Solver`, used as a base for more specific solver classes. The next step in the class hierarchy is `RegularisedSolver`, which implements the basic regularisation workflow. It is a template class, as it does not implement a method for determining the regularisation parameter or the mathematical basis for inversion. Regularisation parameter selection is injected through dedicated selector objects, while the most specific solver classes implement the actual inversion algorithms.

Linear algebra backends are represented by the `Engine` class hierarchy. The base `Engine` class defines the common interface for solving linear systems, while concrete implementations such as `CholeskyEngine` or `SparseInvEngine` provide specific numerical strategies.

Currently, the following inversions are implemented:
 - Biorthogonal basis decomposition (BOB): a method without regularisation based on the `Solver` class.
 - Linear algebraic methods (LAME): methods based on algebraic decomposition of the geometry and regularisation matrices with subsequent series expansion that replaces the algebraic engine. These methods are based on the `RegularisedSolver` class and include the `GevAlgebraic` method based on generalised eigenvalue decomposition and the `SvdAlgebraic` method based on singular value decomposition.
 - Tikhonov regularisation scheme 

Solvers can be used directly or in an iterative process with updated regularisation. This is the basis of the Minimum Fisher Regularisation (MFR) method, which is currently the only implemented iterative regularisation method.
"""
from .base import PearsonSelector, FixedSelector, CholeskySolver
from .bob import Bob, SparseInvSolver
from .lame import GevAlgebraic, SvdAlgebraic, FastSelector
from .mfr import MinimumFisherRegularisation
from .tikhonov import Tikhonov


__all__ = [
	"Bob",
	"CholeskySolver",
	"FastSelector",
	"FixedSelector",
	"GevAlgebraic",
	"MinimumFisherRegularisation",
	"PearsonSelector",
	"SparseInvSolver",
	"SvdAlgebraic",
	"Tikhonov",
]
