Glossary
========

Terminology
-----------

.. glossary::

    detector
        A piece of hardware that converts incident photons to a signal. It can consist of several sensitive elements or channels, e.g. a photodiode array.

    channel
        One sensitive element of a detector producing one signal. For imaging cameras, a channel corresponds to a pixel.

    pixel
        A sensitive element of an imaging camera or an element of a recorded image. Nodes with rectangular cross-section are sometimes called pixels in the literature; this documentation does not use the term in that sense.

    camera
        An enclosure of a detector defining its view and ensuring its survival (shielding, cooling, etc.), e.g. a pinhole camera of a bolometer or soft X-ray system. The term is also used for imaging cameras, such as fast visible cameras, which record two-dimensional images.

    line of sight
    sightline
    chord
        A line approximating the volume observed by a detector channel.

    reconstruction plane
        A plane in which the emissivity is reconstructed, usually the poloidal cross-section for tokamaks. It is either observed directly by the detectors or the lines of sight are projected to it.

    node
        One element of the reconstruction grid described by a local basis function. Toroidally symmetric nodes are volumes with a compact projection to the reconstruction plane.

    voxel
        A term used for nodes in some codes, e.g. Cherab.

    grid
        A set of nodes used for the reconstruction.

    emissivity
        The radiated power per unit volume in the spectral range to which the detectors are sensitive. It is the quantity reconstructed by the inversion.

    geometry matrix
    contribution matrix
    sensitivity matrix
        A matrix describing the contribution of each node to the signal of each channel, with shape (#channels, #nodes). See :doc:`tomography/geometry`.

    retrofit
    forward model
        The signal that would be recorded for a given emissivity, i.e. the product of the geometry matrix and the emissivity.

    regularisation matrix
        A matrix introducing a priori information, usually smoothness, into the inversion. See :doc:`tomography/regularisation`.

    regularisation parameter
        A parameter :math:`\alpha` determining the strength of the a priori information relative to the measured data. See :ref:`tikhonov`.

    phantom
        An artificial emissivity used to create synthetic signals for testing of inversion methods and diagnostic systems. See :doc:`tomography/phantoms`.

    time slice
        Signals of all channels recorded at one time.

    étendue
        The product of the detector area and the solid angle of its view, used to compensate for the finite beam width in the line of sight approximation.

Abbreviations
-------------

.. glossary::

    BoB
        Biorthogonal Basis Decomposition

    GEV
        Generalised Eigenvalue Decomposition

    LAME
        Linear Algebraic Methods

    MFR
        Minimum Fisher Regularisation

    NNLS
        Non-Negative Least Squares

    SVD
        Singular Value Decomposition

    SXR
        Soft X-rays
