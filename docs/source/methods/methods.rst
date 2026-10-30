Inversion methods
=================

Tomotok implements three types of inversion methods in :mod:`tomotok.inversions`. They build on the general background described in :doc:`../tomography/tomography`.

- :doc:`mfr` (MFR) solves the Phillips-Tikhonov regularisation scheme with a nonlinear :term:`regularisation matrix` that is updated iteratively based on the previous solution. It is well established and widely used for under-determined problems.
- :doc:`lame` (LAME) solve the same scheme with a linear regularisation matrix using an algebraic decomposition, which allows fast inversion for multiple values of the :term:`regularisation parameter`.
- :doc:`bob` (BoB) is an unregularised method for over-determined problems, e.g. matrix :term:`cameras <camera>`, which obtains the result by a simple matrix multiplication.

The Phillips-Tikhonov scheme with a given regularisation matrix is also implemented directly by :class:`~tomotok.inversions.tikhonov.Tikhonov`, see :doc:`../tomography/regularisation`.

.. toctree::
    :maxdepth: 1

    mfr
    lame
    bob
