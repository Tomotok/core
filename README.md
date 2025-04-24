# Tomotok
Tomotok is a framework for the tomographic inversion of fusion plasmas, focusing on inversion methods based on discretisation. It is structured as a namespace package to ease implementation on different experiments and across various diagnostics.

# Core
The documentation for the Core can be found on this [link](https://tomotok.github.io/documentation/).

The Core package of Tomotok implements various discretization algorithms that are used for tomographic reconstruction of tokamak plasmas. It is a dependency for specific packages that automate database access for a given fusion experimental device and facilitate routine tomographic computations. Together with the Core package, a simple GUI for result analysis is distributed.

## Inversions
The algorithms accept inputs in the form of `numpy.ndarray` or `scipy.sparse` matrix objects, allowing them to run independently of the rest of the package and promoting interoperability with other codes (e.g., [ToFu](https://tofuproject.github.io/tofu/)).

Currently implemented algorithms:
- Minimum Fisher Regularisation for sparse matrices using `scipy.sparse.linalg.spsolve`
- Minimum Fisher Regularisation for sparse matrices using Cholesky decomposition from scikit-sparse
- SVD linear algebraic inversion for dense matrices
- GEV linear algebraic inversion with optimization for sparse matrices
- Biorthogonal Basis decomposition for dense matrices
- Biorthogonal Basis decomposition optimized for sparse matrices (scipy, cholmod)

## Auxiliary Features
Apart from the main inversion methods, some auxiliary features are also included.

In order to facilitate routine inversion computations, a database interface was designed using template classes. These template classes can load signals, detector view geometry, and magnetic flux reconstruction in the format usually used for tokamak data.

A simple synthetic diagnostic framework is also implemented. It can be used for testing the implemented algorithms. It uses regular rectangular nodes and assumes toroidal symmetry, as it is the simplest case often used for inversions of tokamak plasma radiation.

Implemented auxiliary features:
- Template classes for an automated database interface
- Geometry matrix computation using numerical integration and a single line of sight approximation
- Smoothing matrix computation, both isotropic and anisotropic (based on magnetic flux surfaces)
- Simple phantom model generators (isotropic and anisotropic)
- Other tools for processing

# Citing the Code

"J. Svoboda, J. Cavalier, O. Ficker, M. Imrisek, J. Mlynar and M. Hron, *Tomotok: python package for tomography of tokamak plasma radiation*, Journal of Instrumentation 16.12 (2021): C12015."
[DOI](https://doi.org/10.1088/1748-0221/16/12/c12015)
