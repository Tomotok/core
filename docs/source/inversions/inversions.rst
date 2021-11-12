Inversions
==========

Inversion methods are independent on the rest of the package. Can be used on any data if required inputs are provided. However, this package can provide these only with assumption of toroidal symmetry and for regularly spaced reconstruction nodes of rectangular in reconstruction plane. This was selected as it is the simplest and most widely used option for reconstructions of tokamak plasma. For more detailed description see for example [I1]_, which is a nice overview article on tomography of tokamak plasma.

All methods are using local basis functions and are also referred to as discretisation methods. Currently implemented methods are based on Phillips-Tikhonov regularisation (MFR, LAMe) and biorthogonal basis decomposition (BoB).

Tomographic inversion uses localised measurements to determine spatial distribution of a physical quantity. In case of tokamak plasma, this quantity is usually an electromagnetic radiation emitted by the plasma. There are various parts of the emission spectrum that are used for inversion, depending on the used detector and the physics to be studied. This radiation could be recorded by metallic foil bolometers, photodiode arrays or even fast visible cameras. All of these systems consist of multiple sensitive elements, here referred to as channels. The tomography problem can be written as

.. _tomobase:

.. math::
    \mathbf{f} = \mathbf{T} \cdot \mathbf{g} \,,


where :math:`\mathbf{f}` is the measured signal, :math:`\mathbf{g}` is the discretized emissivity in reconstruction plane and :math:`\mathbf{T}` is a geometry matrix describing the spatial distribution of detectors (also called contribution or sensitivity matrix).

.. toctree::
    :caption: Implemented inversion methods
    :maxdepth: 1

    mfr
    lame
    bob

.. rubric:: References

.. [I1] L. Ingesson, B. Alper, B. Peterson and J.-C. Vallet, Chapter 7: Tomography diagnostics: Bolometry and soft-x-ray detection, Fusion Sci. Technol. 53 (2008) 528.
