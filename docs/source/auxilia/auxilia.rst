Auxiliary functions
===================

The Tomotok package is aiming to focus on inversion methods rather than on providing robust and extensive synthetic diagnostic framework. However, some convenience functions are implemented to make the ease data loading and manipulation, the routines interacting with reconstruction area are implemented only for regular rectangular grids. These are the simplest and most widely used local basis functions for inversion on tokamaks.

Summary
 - Geometry: reconstruction grid definition and geometry matrix computations...
 - IO: classes for automatic database access and data preprocessing
 - Derivative: calculation of derivative matrices on regular grid
 - Phantoms: artificial data generators working with matrices inspired by structure of magnetic flux surfaces.
 - Tools: general purpose functions like post processing methods, visualisations and analysis

.. toctree::
    :caption: Categories of implemented functions
    :maxdepth: 1

    geometry
    io
    derivative
    phantoms
    tools

