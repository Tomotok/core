Minimum Fisher Regularisation
=============================

The Minimum Fisher information Regularisation (MFR) algorithm is based on [M1]_, with further improvements described in [M2]_, and builds on other work such as [M3]_. It is well established and widely used for tokamak plasma tomography.

The algorithm combines the :ref:`Phillips-Tikhonov regularisation <tikhonov>` scheme with minimisation of Fisher information to compensate for the ill-posed nature of the tomography problem. Smoothness is usually the desired property of the reconstruction, and it was shown that the solution with minimal Fisher information is the smoothest one [M1]_. This makes Fisher information a suitable quantity for constructing the :term:`regularisation matrix`.

The inversion solves equation :eq:`tikhonov` with the regularisation matrix constructed in the form of Fisher information

.. math::
   :label: mfr_regularisation

   \mathbf{H} = \sum_i c_i \mathbf{B}^{T}_{i} \cdot \mathbf{w} \cdot \mathbf{B}_{i} \,,

where :math:`\mathbf{B}_i` are numerical derivative matrices, :math:`\mathbf{w}` is a diagonal matrix with weights for individual :term:`nodes <node>` and :math:`c_i` are anisotropy coefficients normalised so that :math:`\sum_i c_i = 1`. This is the general regularisation matrix :eq:`regularisation_matrix` with node weights in the form of Fisher information. The derivative matrices and anisotropy coefficients are described in :doc:`../tomography/regularisation`; the components specific to MFR are described below.

Node weights
   The weights adjust how smooth the solution should be at different nodes: the larger the weight, the smoother the solution. In order for the regularisation matrix to have the form of Fisher information, the weights should be equal to :math:`1/\mathbf{g}`. Since :math:`\mathbf{g}` is the unknown determined by the inversion, the algorithm is nonlinear and is solved iteratively. For the first iteration, an initial estimate of the result can be provided, otherwise the same weight is used for all nodes, which corresponds to linear regularisation. In the following iterations the weights are determined from the previous solution :math:`\mathbf{g}^\mathrm{prev}` as

   .. math::

      w_{ii} = \begin{cases}
         1 / g_i^\mathrm{prev} & \text{if } g_i^\mathrm{prev} > 0 \,, \\
         w_\mathrm{max} & \text{if } g_i^\mathrm{prev} \leq 0 \,,
      \end{cases}

   where :math:`w_\mathrm{max}` is a predefined large number.

Anisotropy coefficients
   The coefficients :math:`c_i` are kept constant during the iterations, so the anisotropy of the regularisation, e.g. preferring smoothness along magnetic flux surfaces, is the same in all iterations, while the node weights adapt the local strength of the regularisation to the solution.

Having constructed the regularisation matrix, the last unknown in :eq:`tikhonov` is the :term:`regularisation parameter` :math:`\alpha`, which is typically found using the :ref:`Pearson test <pearson>`. The solution obtained with this parameter is then used to update the node weights and the regularisation matrix, and the process is repeated until the selected number of iterations is reached. The construction and updating of the regularisation matrix is called the outer loop, while the search for the regularisation parameter is called the inner loop. In practice, the solution barely changes after a few outer iterations and the number of iterations is usually set to 3.

Implementation
--------------

The algorithm is implemented in :class:`~tomotok.inversions.mfr.MinimumFisherRegularisation`. It wraps a :class:`~tomotok.inversions.base.RegularisedInversion` (usually :class:`~tomotok.inversions.tikhonov.Tikhonov`) that performs the inner loop, i.e. determines the regularisation parameter using its regularisation selector and solves :eq:`tikhonov`. A list of inversions, one for each outer iteration, can be provided instead, e.g. to use a :class:`~tomotok.inversions.base.FixedSelector` with a different value of :math:`\alpha` in each iteration and skip the inner loop optimisation. The length of the list has to be equal to ``mfi_num_max``. The outer loop is terminated after ``mfi_num_max`` iterations.

The regularisation matrix :eq:`mfr_regularisation` is constructed from the list of derivative matrices and anisotropy coefficients passed to the call using :func:`~tomotok.regularisations.matrices.weighted_squares`. Instead of a fixed value :math:`w_\mathrm{max}`, the previous solution is clipped from below at ``weight_floor`` (:math:`\varepsilon`) times its maximum before computing the weights

.. math::

   w_{ii} = \frac{1}{\max \left( g_i^\mathrm{prev}, \, \varepsilon \max_j g_j^\mathrm{prev} \right)} \,,

which corresponds to :math:`w_\mathrm{max} = 1 / (\varepsilon \max_j g_j^\mathrm{prev})`. This also bounds the weights for small positive values and keeps the regularisation matrix well conditioned. In the first iteration without an initial estimate, all weights are equal to one.

The speed of the inversion depends mainly on the solver used for :eq:`tikhonov`. The default dense Cholesky decomposition is suitable for small :term:`grids <grid>`, while sparse Cholesky decomposition (e.g. :class:`~tomotok.inversions.solvers.sksparse.SksparseCholesky` from the optional scikit-sparse backend) is significantly faster for large grids.

References
----------

.. [M1] M. Anton et al., "X-ray tomography on the TCV tokamak.", Plasma Phys. Control. Fusion  38.11 (1996): 1849.

.. [M2] J. Mlynar et al., "Current research into applications of tomography for fusion diagnostics." J. Fusion Energy 38.3 (2019): 458-466.

.. [M3] M. Imrisek et al., "Optimization of soft X-ray tomography on the COMPASS tokamak." Nukleonika 61, (2016).
