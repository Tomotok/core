tomotok.geometry
================

.. automodule:: tomotok.geometry
    :no-members:

.. currentmodule:: tomotok.geometry

Reconstruction grids
--------------------

.. autosummary::
    :toctree: generated/

    RegularGrid
    Grid

Geometry matrix
---------------

.. autosummary::
    :toctree: generated/

    sparse_line
    calcam_sparse_line
    dense_line

Lines of sight
--------------

.. autosummary::
    :toctree: generated/

    generate_sightlines
    sightlines.generate_directions
    sightlines.rot_v
    sightlines.rot_h
    sightlines.save_los

Saving and loading
------------------

.. module:: tomotok.geometry.io

.. currentmodule:: tomotok.geometry

.. autosummary::
    :toctree: generated/

    save_sparse_gmat
    load_sparse_gmat
    save_dense_gmat
    load_dense_gmat

Deprecated
----------

.. autosummary::
    :toctree: generated/

    generators.sparse_line_3d
    generators.calcam_sparse_line_3d
    sightlines.generate_los
