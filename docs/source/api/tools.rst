tomotok.tools
=============

.. automodule:: tomotok.tools
    :no-members:

.. currentmodule:: tomotok.tools

Database interface templates
----------------------------

.. autosummary::
    :toctree: generated/

    Diagnostic
    Tokamak

Data containers
---------------

.. autosummary::
    :toctree: generated/

    containers.Signals
    containers.Magnetics
    containers.Reconstruction
    containers.Sightlines

Phantoms
--------

.. automodule:: tomotok.tools.phantoms
    :no-members:

.. currentmodule:: tomotok.tools

.. autosummary::
    :toctree: generated/

    phantoms.regular_elliptical_flux
    phantoms.elliptical_flux
    phantoms.gaussian_on_flux
    phantoms.polar_phase_matrix

Deprecated phantom functions:

.. autosummary::
    :toctree: generated/

    phantoms.iso_psi
    phantoms.gauss
    phantoms.gauss_iso
    phantoms.polar_phase
    phantoms.islands

Checkers
--------

.. autosummary::
    :toctree: generated/

    checkers.AnisotropicDerivativeChecker

HDF utilities
-------------

.. autosummary::
    :toctree: generated/

    hdf.sparse_to_hdf
    hdf.hdf_to_sparse
    hdf.dia_to_hdf
    hdf.hdf_to_dia
    hdf.cs_to_hdf
    hdf.hdf_to_cs

Divertor
--------

.. autosummary::
    :toctree: generated/

    divertor.get_divertor_channels
    divertor.divertor_weighted_matrix
