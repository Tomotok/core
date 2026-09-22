# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
"""
The heart of the package containing the inversion algorithms.

Two orthogonal class hierarchies compose to form each algorithm.

The first defines *what* problem is solved. The most general is `Inversion`, used as a base for the concrete inversion algorithms. The next step in the class hierarchy is `RegularisedInversion`, which implements the basic regularisation workflow. It is a template class, as it does not implement a method for determining the regularisation parameter or the mathematical basis for inversion. Regularisation parameter selection is injected through dedicated `RegularisationSelector` objects, while the most specific subclasses implement the actual inversion algorithms.

The second defines *how* the underlying linear system is solved, represented by the `Solver` class hierarchy in `solvers`. The base `Solver` class defines the common interface for solving linear systems, while concrete implementations such as `CholeskySolver` or `SparseInvSolver` provide specific numerical strategies and are injected into an `Inversion` as a swappable backend.

Currently, the following inversions are implemented:
 - Biorthogonal basis decomposition (BOB): a method without regularisation based on the `Inversion` class.
 - Linear algebraic methods (LAME): methods based on algebraic decomposition of the geometry and regularisation matrices with subsequent series expansion that replaces the solver. These methods are based on the `RegularisedInversion` class and include the `GevAlgebraic` method based on generalised eigenvalue decomposition and the `SvdAlgebraic` method based on singular value decomposition.
 - Tikhonov regularisation scheme

Regularised inversions can be used directly or in an iterative process with updated regularisation. This is the basis of the Minimum Fisher Regularisation (MFR) method, which is currently the only implemented iterative regularisation method.
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
