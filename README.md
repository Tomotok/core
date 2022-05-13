# Tomotok
Framework for tomographic inversion of fusion plasmas focusing on inversion methods based on discretisation. Structured as a namespace package to ease implementation on different experiments and for various diagnostics.

# Core
The documentation for the Core can be found on this [link](https://tomotok.github.io/documentation/).

The Core package of Tomotok implements various discretization algorithm that are used for tomography of tokamak plasma. 
It is required by specific packages that can create automated database access for a given fusion experimental device and ease the routine tomography computation. 
Together with the Core package, simple GUI for result analysis is distributed.

## Inversions
The algorithms take numpy.ndarrays or scipy.sparse matrix objects as input to be able to run independently on the rest of the package in order to promote interoperability with other codes (e.g. [ToFu](https://tofuproject.github.io/tofu/))

Currently implemented algorithms:
 - Minimum Fisher Regularisation for sparse matrices using scipy.sparse.linalg.spsolve
 - Minimum Fisher Regularisation for sparse matrices using cholesky decomposition from scikit.sparse
 - SVD linear algebraic inversion for dense matrices
 - GEV linear algebraic inversion with optimization for sparse matrices
 - Biorthogonal Basis decomposition for dense matrices
 - Biorthogonal Basis decomposition optimized for sparse matrices (scipy, cholmod)

## Auxiliary features

Apart from the main inversion methods some auxiliary features are also included.
In order to make routine computation of inversions a database interface was designed using template classes. These can load signals, detector view geometry and magnetic flux reconstruction in format usually used for tokamak data.

Simple synthetic diagnostic framework is also implemented. It can be used for testing the implemented algorithms. It uses regular rectangular nodes and assumption of toroidal symmetry as it is the simplest case often used for inversions of tokamak plasma radiation.

Implemented auxiliary features:
 - Template classes for automated database interface
 - Geometry matrix computation using numerical integration and single line of sight approximation
 - Smoothing matrix computation, both isotropic and anisotropic (based on magnetic flux surfaces)
 - Simple phantom model generators (isotropic and anisotropic)
 - Other tools for processing 

## Graphical user interface

Simple graphical user interface for visualisation and post-processing of tomography results is included in the Core package. It is based on modular system of windows. It uses main window to spawn child windows for analysis to customize displayed information based on user needs. 

# Citing the Code

"J. Svoboda, J. Cavalier, O.Ficker, M. Imrisek, J. Mlynar and M. Hron, *Tomotok: python package for tomography of tokamak plasma radiation*, Journal of Instrumentation 16.12 (2021): C12015."
[DOI](https://doi.org/10.1088/1748-0221/16/12/c12015)
