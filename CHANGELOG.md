Version 2.0 (DD.MM.2024)
========================

Changes
-------
 - Default MFR class uses cho_factor and cho_solve from scipy
 - reworked format of line of sights, updated gmat computation functions
 - line of sight saving to a single json file
 - new function for calculation of anisotropic derivative matrix

New Features
------------
 - tutorial for checking anisotropic derivative matrix
 - tests for line of sight saving and loading
 - tests for derivative matrix computation

Fixes
-----
 - updated derivative matrix checker

Version 1.3 (21.07.2023)
=======================

Changes
-------
 - minor improvements of documentation structure

New Features
------------
 - added more methods for derivative matrix computation
 - added new geometry matrix generator for start and end points

Fixes
-----
 - step parameter name in geometry matrix generator for calcam
 - keywords passing to BOB
 - MFR accepts 1d signal
 - tomotok article citation info

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
