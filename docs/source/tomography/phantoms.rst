Testing with phantoms
=====================

It is common practice to test the performance of a tomographic system, or of an inversion method, using phantom emissivity models. A phantom is used to create synthetic signals using the forward model, i.e. by multiplying the flattened phantom by the geometry matrix. The synthetic signals, optionally with added noise, are inverted and the result can be directly compared with the phantom. The phantoms usually represent typical plasma phenomena expected to be observed and can be based on experience, theoretical models or numerical simulations. This helps to design the system for its intended use and to understand its limitations, see e.g. [P1]_ for the ITER soft X-ray system.

Artificial emissivity can be created with the functions in :mod:`tomotok.tools.phantoms`. The emissivity can be isotropic or based on provided magnetic flux values, e.g. artificial flux surfaces of elliptical shape created by :func:`~tomotok.tools.phantoms.elliptical_flux`.

When the results are compared with the phantom, a measure of the difference such as the root mean square error should be used. The Pearson :math:`\chi^2` test is not suitable for this purpose, since the :ref:`regularisation parameter <pearson>` is usually selected so that :math:`\chi^2` is close to one.

Peak method
-----------

The peak method tests the coverage of a detector layout in different parts of the reconstruction plane [P2]_. Narrow Gaussian peaks are scattered over the reconstruction plane and synthetic signals are created for each peak separately. The peaks should span several nodes to be compatible with algorithms imposing smoothness, such as MFR. The estimated position and height of the peak in each reconstruction are then compared with the parameters of the phantom, which allows the coverage to be quantified in different areas.

A peak phantom can be created with :func:`~tomotok.tools.phantoms.gaussian_on_flux` by passing the distance from the peak position instead of magnetic flux values:

.. code-block:: python

    import numpy as np
    from tomotok.geometry import RegularGrid
    from tomotok.tools.phantoms import gaussian_on_flux

    grid = RegularGrid(40, 60, (0.3, 0.7), (-0.3, 0.3))
    r, z = grid.center_mesh
    distance = np.hypot(r - 0.5, z - 0.1)  # peak centred at R = 0.5, z = 0.1
    phantom = gaussian_on_flux(distance, width=1e-3, limit=0.05, limit_width=0.02)
    signals = gmat @ phantom.flatten()  # gmat computed for the same grid

References
----------

.. [P1] D. Colette et al., "Modeling a low voltage ionization chamber based tomography system on ITER," Rev. Sci. Instrum. 91 (2020): 073504.

.. [P2] J. Svoboda et al., "Conceptual design of tomographic soft X-ray detectors for COMPASS-U tokamak," Fusion Eng. Des. 168 (2021): 112656.
