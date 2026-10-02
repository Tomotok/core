Derivative matrix
=================

Derivative matrices are used to construct regularisation matrices for the regularised inversion methods :doc:`../inversions/mfr` and :doc:`../inversions/lame`. Smoothness is usually the desired property of the reconstruction, as most physical properties of the plasma are expected to change smoothly, and regularisation matrices built from discretised derivatives penalise solutions that are not smooth. The description below draws on section 1.2.1 of [I2]_.

Numerical schemes and directions
--------------------------------

The derivative matrices can employ different finite difference schemes (forward, backward, central or second order) and directions. The choice of scheme and direction has an impact on the result [M3]_. To overcome this, a combination of schemes or directions can be used. A typical choice is to combine the forward scheme in all four basic directions, or additionally in the diagonal directions, totalling eight derivative matrices, see :func:`~tomotok.regularisations.derivatives.all_direction_derivative_matrices`.

If the direction is the same for the entire reconstruction grid, the derivative matrix is called isotropic. It is also possible to use different directions in different parts of the grid, creating an anisotropic derivative matrix. A typical example is a matrix that introduces smoothness along magnetic flux surfaces. The function :func:`~tomotok.regularisations.derivatives.anisotropic_derivative_matrix` uses the direction of the flux gradient in each node and distributes the contribution between the two neighbouring nodes closest to the desired direction. :func:`~tomotok.regularisations.derivatives.standard_anisotropic_derivative_matrices` provides four such matrices, in the directions parallel and perpendicular to the flux surfaces, each in both orientations (clockwise and counterclockwise). The :class:`~tomotok.tools.checkers.AnisotropicDerivativeChecker` can be used to visually check the computed matrices.

Regularisation matrix
---------------------

The derivative matrices :math:`\mathbf{B}_i` are combined into the regularisation matrix as

.. math::
    \mathbf{H} = \sum_i c_i \mathbf{B}^{T}_{i} \cdot \mathbf{w} \cdot \mathbf{B}_{i} \,,

where :math:`c_i` are anisotropy coefficients, normalised so that :math:`\sum_i c_i = 1` (only their ratios matter), and :math:`\mathbf{w}` is a diagonal matrix with node weights, which is equal to the identity matrix for linear regularisation, see :func:`~tomotok.regularisations.matrices.weighted_squares`. Increasing a coefficient :math:`c_i` relative to the others requires the result to be smoother in the direction represented by the respective derivative matrix. In combination with a magnetic equilibrium reconstruction and anisotropic derivative matrices, the coefficients can be used to prefer solutions that are smoother along the magnetic flux surfaces and steeper perpendicular to them. This is usually referred to as anisotropic regularisation.

Nodes outside the region of interest, e.g. outside the vessel, can be removed from the derivative matrices using the ``mask`` parameter. The ``compensate_edges`` option, enabled by default, modifies the diagonal so that the sum of each row is zero, i.e. a constant emissivity is not penalised, also in the nodes at the edge of the grid or mask. Without it, the edge rows penalise the value of the emissivity itself, as if it were zero outside the grid. With compensation, the regularisation matrix is singular, with the constant vector as its null space, which has to be taken into account for :doc:`../inversions/lame`.

Implementation
--------------

.. automodule:: tomotok.regularisations.derivatives
    :members:
    :show-inheritance:
    :noindex:
