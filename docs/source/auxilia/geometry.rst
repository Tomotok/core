Geometry
========
Handles the geometry of the inversion task. This includes the definition of reconstruction grids, the creation of artificial lines of sight and the computation of geometry (sensitivity/contribution) matrices. Currently, only algorithms using the single line of sight approximation are implemented. The description below draws on the introduction to chapter 1 and on section 1.1 of [I2]_.

Discretisation
--------------

For detectors measuring intensity, the power deposited to the i-th detector can be expressed as

.. math::
    f_i = \int T_i(\mathbf{r}) \, g(\mathbf{r}) \, \mathrm{d}\mathbf{r} \,,

where :math:`T_i` describes the geometry of the volume observed by the i-th detector and :math:`g(\mathbf{r})` is the emissivity in the spectral range to which the detector is sensitive. The emissivity is expanded into a series of basis functions :math:`b_j(\mathbf{r})` with coefficients :math:`g_j`

.. math::
    g(\mathbf{r}) = \sum_{j=1}^{N_\mathrm{n}} b_j(\mathbf{r}) \, g_j \,.

Substituting the expansion and interchanging the order of summation and integration gives the matrix form :math:`\mathbf{f} = \mathbf{T} \cdot \mathbf{g}` of the :ref:`tomography problem <tomobase>`, where the elements of the geometry matrix describe the contribution of each basis function to the signal of a given detector

.. math::
    :label: gmat_element

    T_{ij} = \int T_i(\mathbf{r}) \, b_j(\mathbf{r}) \, \mathrm{d}\mathbf{r} \,.

The most commonly used basis functions for tokamak plasma tomography are local basis functions. They represent volumes, called reconstruction nodes, with a compact projection to the poloidal plane, and a set of nodes is called a grid. The nodes are usually assumed to be toroidally symmetric, which is the most common assumption for general applications. For special applications, such as edge plasma tomography, toroidal symmetry is not valid and the nodes have to be shaped differently, e.g. following the magnetic field lines in the toroidal direction (helicoidal symmetry) [B2]_. Nodes with rectangular cross-section are often referred to as pixels in the literature. In this documentation, the term pixel refers exclusively to a sensitive element of a camera or an element of a recorded image to avoid confusion.

Tomotok implements toroidally symmetric nodes with rectangular projection to the poloidal plane on a regular grid, see :class:`~tomotok.geometry.RegularGrid`. With a sufficiently large number of nodes, no prior knowledge about the plasma shape is required, and information about the shape can be introduced by the regularisation instead (see :doc:`derivative`). Nodes are ordered row by row, i.e. the flattened emissivity corresponds to an array with shape ``(nz, nr)``.

Geometry matrix
---------------

The geometry matrix can in principle be calculated analytically. However, this is rather complicated and a simplified approach is used in practice. One approach approximates the viewing cone of each detector by a line of sight and compensates for the finite beam width using étendue [I1]_. An alternative is ray-tracing with randomised sampling.

Line of sight approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This approach uses a single line to approximate the view of a detector. Typically, the line starts in the middle of the detector surface and goes through the centre of the pinhole or collimator, although the situation can be more complex in some cases [G2]_. The line is usually called a line of sight, sightline or chord. In this approximation, the geometrical term :math:`T_i(\mathbf{r})` is equal to one on the line of sight and zero elsewhere, which turns the volume integral :eq:`gmat_element` into a line integral. For local basis functions, the element of the geometry matrix is then equal to the length of the line of sight inside the j-th node

.. math::
    T_{ij} = \int b_j(l) \, \mathrm{d}l = \int_{b_j} \mathrm{d}l \,.

The approximation has some requirements and limitations:

- The observed emissivity should not change significantly in the direction perpendicular to the line of sight within the volume observed by each detector [G2]_.
- Differences in the geometry of detector apertures cannot be distinguished, i.e. the result is the same regardless of pinhole or collimator size.
- The geometry matrix has units of length. A compensation by étendue, approximated e.g. by the product of the detector area and the solid angle of its view, is necessary if signals in Watts are to be used.

In Tomotok, this approach is implemented by :func:`~tomotok.geometry.generators.sparse_line`. Each line of sight is divided into short segments of length ``step`` and the length of the segments in each node is accumulated. A central column blocking the view can be taken into account using the ``rmin`` parameter. The étendue compensation is not included and has to be applied to the signals or to the geometry matrix by the user.

Ray-tracing
^^^^^^^^^^^

Ray-tracing casts a large number of rays from a detector with randomly selected origins and directions and follows their paths through the reconstruction grid. Each ray is handled similarly to a line of sight and the result is again a geometry matrix. A significant advantage of this approach is that it accounts for the size of the pinhole or collimator, so the requirement on emissivity not changing between neighbouring lines of sight is not needed, assuming an appropriate grid is used. If the rays are normalised by the solid angle, no compensation by étendue is needed. Modern ray-tracing codes such as Raysect and Cherab can also account for reflections on plasma-facing components, which is typically not needed for soft X-rays but can be useful e.g. for visible cameras [G1]_.

Ray-tracing is not implemented in Tomotok, but geometry matrices computed by external codes can be used with all inversion methods, provided that the order of nodes matches the reconstruction grid.

.. automodule:: tomotok.geometry.generators
    :members:
    :undoc-members:
    :noindex:

Reconstruction grids
--------------------

.. autoclass:: tomotok.geometry.RegularGrid
    :members:
    :noindex:

Sightlines
----------

.. automodule:: tomotok.geometry.sightlines
    :members:
    :undoc-members:
    :noindex:

References
----------

.. [G1] M. Carr et al., "Description of complex viewing geometries of fusion tomography diagnostics by ray-tracing," Rev. Sci. Instrum. 89 (2018): 083506.

.. [G2] L.C. Ingesson et al., "Projection-space methods to take into account finite beam-width effects in two-dimensional tomography algorithms," J. Opt. Soc. Am. A 16 (1999): 17-27.
