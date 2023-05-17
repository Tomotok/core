Database access (IO)
====================
This module contains classes for automatic database access and data preprocessing. The fundamental class is called Dsystem and describes geometry and database access for a given diagnostic system. This presents the basic set of functions that can be expanded by sub-classing. A subclass Tokamak that can handle loading and interpolating magnetic data in a format usual for tokamaks is implemented.

The convention for coordinates order is 
 - for reconstructions (time, channel, node) or (time, channel, vertical node coordinate, horizontal node coordinate)
 - for matrix cameras (time, vertical pixel coord, horizontal pixel coord, vertical node coordinate, horizontal node coordinate)

Tokamak magnetics are expected to be computed on regular grid. This is typical output of EFIT. Currently they are stored in dicts with keys `values`, `time`, `z`, `r`.

Diagnostic system
-----------------
Basic interface with detectors. Handles signals loading and calibrations, cameras geometries...

.. autoclass:: tomotok.core.io.Dsystem
    :members:
    :show-inheritance:
    :noindex:


Tokamak
-------
Adds methods for handling magnetics data in format typical for tokamak equilibrium.

.. autoclass:: tomotok.core.io.Tokamak
    :members:
    :show-inheritance:
    :noindex:
