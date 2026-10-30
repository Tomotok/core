Regularisation
==============

Most tomographic systems on tokamaks lead to an under-determined problem, see :doc:`problem`. A priori information has to be added to select a unique solution, which is usually done by the Phillips-Tikhonov regularisation with a :term:`regularisation matrix` constructed from derivative matrices. The derivative and regularisation matrices are provided by :mod:`tomotok.regularisations`.

.. _tikhonov:

Phillips-Tikhonov regularisation
--------------------------------

Phillips-Tikhonov regularisation is a mathematical recipe for solving ill-posed problems. It adds a term to the problem that prefers solutions with certain properties. Applying it to the :ref:`tomography problem <tomobase>` results in [I1]_

.. math::
   :label: tikhonov

   ( \mathbf{T}^T \cdot \mathbf{T} + \alpha \mathbf{H} ) \cdot \mathbf{g} = \mathbf{T}^T \cdot \mathbf{f} \,,

where :math:`\mathbf{H} \in \mathbb{R}^{N_\mathrm{n}, N_\mathrm{n}}` is the regularisation matrix and :math:`\alpha` is the :term:`regularisation parameter` determining the strength of the a priori information relative to the measured data. The standard Tikhonov regularisation uses the identity matrix, penalising solutions with a large norm. This is not suitable for plasma radiation tomography and regularisation matrices constructed from discretised derivatives are used instead, since most physical properties of the plasma are expected to change smoothly. The signals and the :term:`geometry matrix` are normalised by the expected errors :math:`\sigma_i` of individual :term:`channels <channel>` before the inversion, so that channels with lower uncertainty have larger weight. The geometry matrix and the signals in :eq:`tikhonov` therefore denote the normalised quantities :math:`\mathrm{diag}(1/\sigma_i) \cdot \mathbf{T}` and :math:`\mathrm{diag}(1/\sigma_i) \cdot \mathbf{f}`, and the value of :math:`\alpha` refers to this normalised problem.

In Tomotok, equation :eq:`tikhonov` is solved by :class:`~tomotok.inversions.Tikhonov`. The linear system is solved by an interchangeable :class:`~tomotok.inversions.solvers.Solver` backend (dense or sparse Cholesky decomposition, non-negative least squares and others, see :mod:`tomotok.inversions.solvers`).

Derivative matrices
-------------------

The derivative matrices can employ different finite difference schemes (forward, backward, central or second order) and directions. The choice of scheme and direction has an impact on the result [M3]_. To overcome this, a combination of schemes or directions can be used. A typical choice is to combine the forward scheme in all four basic directions, or additionally in the diagonal directions, totalling eight derivative matrices, see :func:`~tomotok.regularisations.all_direction_derivative_matrices`.

If the direction is the same for the entire reconstruction :term:`grid`, the derivative matrix is called isotropic. It is also possible to use different directions in different parts of the grid, creating an anisotropic derivative matrix. A typical example is a matrix that introduces smoothness along magnetic flux surfaces. The function :func:`~tomotok.regularisations.anisotropic_derivative_matrix` uses the direction of the flux gradient in each :term:`node` and distributes the contribution between the two neighbouring nodes closest to the desired direction. :func:`~tomotok.regularisations.standard_anisotropic_derivative_matrices` provides four such matrices, in the directions parallel and perpendicular to the flux surfaces, each with the forward and backward scheme. For the parallel direction, the schemes correspond to the counterclockwise and clockwise orientation along the flux surfaces; for the perpendicular direction, to the direction of increasing and decreasing flux. The :class:`~tomotok.tools.checkers.AnisotropicDerivativeChecker` can be used to visually check the computed matrices.

Regularisation matrix
---------------------

The derivative matrices :math:`\mathbf{B}_i` are combined into the regularisation matrix as

.. math::
    :label: regularisation_matrix

    \mathbf{H} = \sum_i c_i \mathbf{B}^{T}_{i} \cdot \mathbf{w} \cdot \mathbf{B}_{i} \,,

where :math:`c_i` are anisotropy coefficients, normalised so that :math:`\sum_i c_i = 1` (only their ratios matter), and :math:`\mathbf{w}` is a diagonal matrix with node weights, which is equal to the identity matrix for linear regularisation, see :func:`~tomotok.regularisations.weighted_squares`. Increasing a coefficient :math:`c_i` relative to the others requires the result to be smoother in the direction represented by the respective derivative matrix. In combination with a magnetic equilibrium reconstruction and anisotropic derivative matrices, the coefficients can be used to prefer solutions that are smoother along the magnetic flux surfaces and steeper perpendicular to them. This is usually referred to as anisotropic regularisation. It is the most common choice for tokamak plasma, motivated by theory, even though it is only one of many possible combinations.

Nodes outside the region of interest, e.g. outside the vessel, can be removed from the derivative matrices using the ``mask`` parameter, a boolean array with shape ``(nz, nr)`` that is ``True`` for the nodes to keep. The geometry matrix has to be reduced in the same way, e.g. ``gmat[:, mask.ravel()]``, and the result of the inversion then contains values only for the kept nodes. The ``compensate_edges`` option, enabled by default, modifies the diagonal so that the sum of each row is zero, i.e. a constant :term:`emissivity` is not penalised, also in the nodes at the edge of the grid or mask. Without it, the edge rows penalise the value of the emissivity itself, as if it were zero outside the grid. With compensation, the regularisation matrix is singular, with the constant vector as its null space, which has to be taken into account for :doc:`../methods/lame`.

.. _pearson:

Regularisation parameter
------------------------

There are multiple methods for determining the regularisation parameter, see [I1]_ or [L1]_. A method that has proved reliable is based on the Pearson :math:`\chi^2` test. It determines the optimal value of :math:`\alpha` by comparing the misfit of the reconstruction with the estimated error of the measured data

.. math::
   \chi^2 = \frac{1}{N_\mathrm{c}} \sum_{i=1}^{N_\mathrm{c}} \left( \frac{ f_i - \left( \mathbf{T} \cdot \mathbf{g} \right)_i }{\sigma_i} \right)^2 \,,

where :math:`f_i` is the signal measured by the i-th channel and :math:`\sigma_i` is its expected error. The optimal value is considered to be one [M2]_, which means that the misfit of the reconstruction is approximately the same as the estimated error of the measured data.

This test is implemented in :class:`~tomotok.inversions.PearsonSelector`, which is the default regularisation parameter selector. It minimises :math:`|\chi^2 - 1|` using a bounded search algorithm combined with Brent's method (:func:`scipy.optimize.minimize_scalar`) [I2]_ [I3]_. The search is conducted on a logarithmic scale within bounds representing values of :math:`\alpha` that are considered too small or too large, by default :math:`10^{-30} \leq \alpha \leq 10^{10}`. The main disadvantage of this approach is its computational cost, as one inversion of :eq:`tikhonov` is required for each tested value of :math:`\alpha`.

Repeated optimisation can be avoided by using :class:`~tomotok.inversions.FixedSelector` with a regularisation parameter provided by the user. This utilises the fact that optimisation on similar data (e.g. consecutive :term:`time slices <time slice>`) provides similar values of the regularisation parameter. Linear algebraic methods additionally provide a fast estimate based on their decomposition, see :doc:`../methods/lame`.

References
----------

.. [I2] G. E. Forsythe, M. A. Malcolm and C. B. Moler, Computer Methods for Mathematical Computations, Prentice-Hall (1977).

.. [I3] R. P. Brent, Algorithms for Minimization without Derivatives, Prentice-Hall (1973).
