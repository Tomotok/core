Version 1.1 (*.05.2022)
=====================

Changes
-------
 - renamed MFR parameter danis to aniso
 - renamed presolve method of linear solvers to decompose
 - unified naming of variables, parameters and methods for MFR and LAME

New Features
------------
 - MFR returns inversion statistics
 - MFR supports setting tolerance of optimal regularisation parameter value
 - simple sparse optimization of GevFastAlgebraic
 - RegularGrid supports rlim, zlim and corners properties
 - improved input checks of LAME and MFR methods

Fixes
-----
 - correct documentation link in readme
 - added namespace declaration to setup.py
