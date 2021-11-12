Database access (IO)
====================

This module contains classes for automatic database access and data preprocessing. The fundamental class is called Dsystem and describes geometry and database access for a given diagnostic system.

Results saved in hdf files with coordinates in following orders (time, channel, node) or (time, channel, vertical node coordinate, horizontal node coordinate)

for matrix cameras (time, vertical pixel coord, horizontal pixel coord, vertical node coordinate, horizontal node coordinate)

Tokamak magnetics expect to be computed on regular grid. These are saved in npz with dict keys `values`, `time`, `z`, `r`

Diagnostic system
-----------------
Basic interface with detectors. Handles signals loading and calibrations, cameras geometries...

.. autoclass:: tomotok.core.io.Dsystem
    :members:
    :show-inheritance:
    :noindex:


Tokamak
-------
Adds methods for handling magnetics data in format used on tokamaks

.. autoclass:: tomotok.core.io.Tokamak
    :members:
    :show-inheritance:
    :noindex:
