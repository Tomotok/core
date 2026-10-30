Quick start
===========

The following example shows a complete inversion on synthetic data: definition of a reconstruction grid, computation of a geometry matrix for two pinhole cameras, creation of synthetic signals from a phantom and their inversion using :doc:`methods/mfr`.

.. code-block:: python

    import numpy as np

    from tomotok.geometry import RegularGrid
    from tomotok.geometry.generators import sparse_line
    from tomotok.geometry.sightlines import generate_sightlines
    from tomotok.inversions import MinimumFisherRegularisation, Tikhonov
    from tomotok.regularisations import all_direction_derivative_matrices
    from tomotok.tools.phantoms import elliptical_flux, gaussian_on_flux

    # reconstruction grid with 30 x 40 nodes
    grid = RegularGrid(30, 40, rlims=(0.3, 0.7), zlims=(-0.3, 0.3))

    # two pinhole cameras with 20 lines of sight each
    cameras = [
        generate_sightlines(pinhole=(0.9, 0, 0), num=(20, 1), fov=(60, 0), axis=(-1, 0, 0), length=1.5),
        generate_sightlines(pinhole=(0.5, 0, 0.5), num=(20, 1), fov=(60, 0), axis=(0, 0, -1), length=1.5),
    ]
    starts = np.vstack([c[0] for c in cameras])
    ends = np.vstack([c[1] for c in cameras])
    gmat = sparse_line(starts, ends, grid)  # geometry matrix with shape (#channels, #nodes)

    # synthetic signals from a phantom with added noise
    phantom = gaussian_on_flux(elliptical_flux(grid.nr, grid.nz), width=0.2)
    signals = gmat @ phantom.flatten()
    errors = 0.02 * signals.max()
    signals = signals + np.random.default_rng(0).normal(0, errors, signals.shape)

    # Minimum Fisher Regularisation with Tikhonov inner loop
    derivatives = all_direction_derivative_matrices(grid)
    mfr = MinimumFisherRegularisation(Tikhonov(), mfi_num_max=3)
    emissivity, stats = mfr(signals, gmat, derivatives, errors)
    emissivity = emissivity.reshape(grid.shape)  # (nz, nr) array

The ``stats`` list contains the regularisation parameter and the result of the :ref:`Pearson test <pearson>` for each MFR iteration. The theoretical background, including the computation of geometry matrices, the regularisation and testing with phantoms, is described in :doc:`tomography/tomography`, and the other inversion methods in :doc:`methods/methods`.
