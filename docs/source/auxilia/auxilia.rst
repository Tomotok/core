Auxiliary functions
===================

As the package is intended to focus on inversion methods rather than on providing robust and extensive synthetic diagnostic framework, the routines described below are implemented only for regular rectangular grids. These are the simplest and most widely used local basis functions for inversion on tokamaks.

Geometry
---------
Handles geometry of the inversion task. This includes definition of reconstruction grid, creating artificial geometry and computation of geometry (sensitivity/contribution) matrices. Currently only algorithms using single line of sight approximation are implemented.

Database Access (IO)
--------------------
This module contains classes for automatic database access and data preprocessing. The fundamental class is called Dsystem and describes geometry and database access for a given diagnostic system. This presents the basic set of functions that can be expanded by sub-classing. A subclass Tokamak that can handle loading and interpolating magnetic data in a format usual for tokamaks is implemented.

Derivative
----------
Used in regularised inversion methods MFR and LAME. Various schemes for computation are used. Anisotropic derivatives can be computed using magnetics data.

Phantoms
--------
Artificial data generators working with matrices inspired by structure of magnetic flux surfaces.

Tools
-----
Contains general purpose functions like post processing methods, visualisations and analysis.
