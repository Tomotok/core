Geometry matrix
===============

The geometry matrix can in principle be calculated analytically. However, this is rather complicated and a simplified approach is used in practice. One approach approximates the viewing cone of each detector by a line of sight and compensates for the finite beam width using étendue [I1]_. An alternative is ray-tracing with randomised sampling.

Line of sight approximation
---------------------------

This approach uses a single line to approximate the view of a detector. Typically, the line starts in the middle of the detector surface and goes through the centre of the pinhole or collimator, although the situation can be more complex in some cases [G2]_. The line is usually called a line of sight, sightline or chord. In this approximation, the geometrical term :math:`T_i(\mathbf{r})` is equal to one on the line of sight and zero elsewhere, which turns the volume integral :eq:`gmat_element` into a line integral. For local basis functions, the element of the geometry matrix is then equal to the length of the line of sight inside the j-th node

.. math::
    T_{ij} = \int b_j(l) \, \mathrm{d}l = \int_{b_j} \mathrm{d}l \,.

The approximation has some requirements and limitations:

- The observed emissivity should not change significantly in the direction perpendicular to the line of sight within the volume observed by each detector [G2]_.
- Differences in the geometry of detector apertures cannot be distinguished, i.e. the result is the same regardless of pinhole or collimator size.
- The geometry matrix has units of length. A compensation by étendue, approximated e.g. by the product of the detector area and the solid angle of its view, is necessary if signals in Watts are to be used.

In Tomotok, this approach is implemented by :func:`~tomotok.geometry.generators.sparse_line`. Each line of sight is divided into short segments of length ``step`` and the length of the segments in each node is accumulated. A central column blocking the view can be taken into account using the ``rmin`` parameter. The étendue compensation is not included and has to be applied to the signals or to the geometry matrix by the user.

Artificial lines of sight of pinhole cameras, e.g. for testing of detector layouts, can be created by :func:`~tomotok.geometry.sightlines.generate_sightlines`, and geometry matrices can be saved and loaded together with the grid using the functions in :mod:`tomotok.geometry.io`.

Ray-tracing
-----------

Ray-tracing casts a large number of rays from a detector with randomly selected origins and directions and follows their paths through the reconstruction grid. Each ray is handled similarly to a line of sight and the result is again a geometry matrix. A significant advantage of this approach is that it accounts for the size of the pinhole or collimator, so the requirement on emissivity not changing between neighbouring lines of sight is not needed, assuming an appropriate grid is used. If the rays are normalised by the solid angle, no compensation by étendue is needed. Modern ray-tracing codes such as Raysect and Cherab can also account for reflections on plasma-facing components, which is typically not needed for soft X-rays but can be useful e.g. for visible cameras [G1]_.

Ray-tracing is not implemented in Tomotok, but geometry matrices computed by external codes can be used with all inversion methods, provided that the order of nodes matches the reconstruction grid.

References
----------

.. [G1] M. Carr et al., "Description of complex viewing geometries of fusion tomography diagnostics by ray-tracing," Rev. Sci. Instrum. 89 (2018): 083506.

.. [G2] L.C. Ingesson et al., "Projection-space methods to take into account finite beam-width effects in two-dimensional tomography algorithms," J. Opt. Soc. Am. A 16 (1999): 17-27.
