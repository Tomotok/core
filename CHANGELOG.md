Version 1.3 ()
==============

Fixes
-----
 - keywords passing to BOB
 - MFR accepts 1d signal

Version 1.2 (14.10.2022)
========================

New Features
------------
 - BOB decomposition supports passing keywords to the inversion solver

Fixes
-----
 - corrected dependencies versions
 - replaced f strings by format method

Version 1.1 (12.05.2022)
========================

Changes
-------
 - renamed MFR parameter danis to aniso
 - renamed presolve method of linear solvers to decompose
 - unified naming of variables, parameters and methods for MFR and LAME
 - SimpleBob deprecated by general class Bob taking basis as parameter

New Features
------------
 - MFR returns inversion statistics
 - MFR supports setting tolerance of optimal regularisation parameter value
 - BOB supports regularisation
 - New subclasses of BOB using sparse inverse and Cholesky decomposition
 - simple sparse optimization of GevFastAlgebraic
 - RegularGrid supports rlim, zlim and corners properties
 - improved input checks of LAME and MFR methods

Fixes
-----
 - correct documentation link in readme
 - added namespace declaration to setup.py
