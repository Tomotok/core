Tomography problem
==================

Tomographic inversion uses line-integrated (or volume-integrated) measurements to determine the spatial distribution of a physical quantity. In the case of tokamak plasma, this quantity is usually electromagnetic radiation emitted by the plasma. Various parts of the emission spectrum are used for inversion, depending on the :term:`detector` and the physics to be studied. This radiation can be recorded by metallic foil bolometers, photodiode arrays or even fast visible :term:`cameras <camera>`. All of these systems consist of multiple sensitive elements, here referred to as :term:`channels <channel>`.

Discretisation
--------------

For detectors measuring intensity, the power deposited to the i-th detector can be expressed as

.. math::
    f_i = \int T_i(\mathbf{r}) \, g(\mathbf{r}) \, \mathrm{d}\mathbf{r} \,,

where :math:`T_i` describes the geometry of the volume observed by the i-th detector and :math:`g(\mathbf{r})` is the :term:`emissivity` in the spectral range to which the detector is sensitive. The emissivity is expanded into a series of basis functions :math:`b_j(\mathbf{r})` with coefficients :math:`g_j`

.. math::
    g(\mathbf{r}) = \sum_{j=1}^{N_\mathrm{n}} b_j(\mathbf{r}) \, g_j \,.

Substituting the expansion and interchanging the order of summation and integration gives the matrix form of the tomography problem

.. _tomobase:

.. math::
    \mathbf{f} = \mathbf{T} \cdot \mathbf{g} \,,

where :math:`\mathbf{f} \in \mathbb{R}^{N_\mathrm{c}}` is the measured signal of :math:`N_\mathrm{c}` channels, :math:`\mathbf{g} \in \mathbb{R}^{N_\mathrm{n}}` is the discretised emissivity in :math:`N_\mathrm{n}` :term:`nodes <node>` of the :term:`reconstruction plane` and :math:`\mathbf{T} \in \mathbb{R}^{N_\mathrm{c}, N_\mathrm{n}}` is the :term:`geometry matrix` describing the spatial sensitivity of the detectors (also called contribution or sensitivity matrix). Its elements describe the contribution of each basis function to the signal of a given detector

.. math::
    :label: gmat_element

    T_{ij} = \int T_i(\mathbf{r}) \, b_j(\mathbf{r}) \, \mathrm{d}\mathbf{r} \,.

Their computation is described in :doc:`geometry`. The product :math:`\mathbf{T} \cdot \mathbf{g}` is called the :term:`retrofit` or forward model; it represents the signal that would be recorded for emissivity :math:`\mathbf{g}`.

Reconstruction nodes
--------------------

All methods implemented in Tomotok use local basis functions, i.e. disjoint compact basis functions that are equal to one in a given node and zero elsewhere, which is why they are also referred to as discretisation or series expansion methods. The basis functions represent volumes, called reconstruction nodes, with a compact projection to the poloidal plane, and a set of nodes is called a :term:`grid`. The main advantage of local basis functions is their universality and simple implementation. With a sufficiently large number of nodes, no prior knowledge about the plasma shape or the position of the magnetic axis is required and a regular grid can be used. Information about the plasma shape can instead be introduced by the regularisation, e.g. by anisotropic regularisation.

The nodes are usually assumed to be toroidally symmetric, which is the most common assumption for general applications. For special applications, such as edge plasma tomography, toroidal symmetry is not valid and the nodes have to be shaped differently, e.g. following the magnetic field lines in the toroidal direction (helicoidal symmetry) [B2]_. Nodes with rectangular cross-section are often referred to as :term:`pixels <pixel>` in the literature. In this documentation, the term pixel refers exclusively to a sensitive element of a camera or an element of a recorded image to avoid confusion.

Tomotok implements toroidally symmetric nodes with rectangular projection to the poloidal plane on a regular grid, see :class:`~tomotok.geometry.RegularGrid`. Nodes are ordered row by row, i.e. the flattened emissivity corresponds to an array with shape ``(nz, nr)``. The first row corresponds to the lowest values of :math:`z`, so the reconstruction has to be plotted e.g. with ``origin='lower'`` in :func:`matplotlib.pyplot.imshow` or using :func:`matplotlib.pyplot.pcolormesh` with the node borders of the grid.

Under- and over-determined problems
-----------------------------------

Tomography is typically used with data from linear detector arrays, i.e. systems consisting of individual detectors placed next to each other on a line or small detector arrays creating a fan-like configuration of :term:`lines of sight <line of sight>`. These usually have significantly fewer channels than nodes used for the inversion (:math:`N_\mathrm{c} < N_\mathrm{n}`), which makes the problem under-determined with an infinite number of possible solutions. A priori information has to be added to select a unique solution. This is usually done with a :term:`regularisation matrix`, which commonly consists of discretised derivatives and penalises solutions that are not smooth, see :doc:`regularisation`. Such methods are implemented in :doc:`../methods/mfr` and :doc:`../methods/lame`.

Systems based on cameras or other matrix detectors can have more pixels than there are nodes (:math:`N_\mathrm{c} \geq N_\mathrm{n}`), making the problem over-determined. These systems do not necessarily require regularisation to obtain a unique solution, and unregularised methods such as :doc:`../methods/bob` can be used. Regularisation can still improve the solution if the camera view does not cover the vessel well.
