Linear Algebraic Methods
========================

The linear algebraic methods (LAME) were investigated and improved as a high-speed alternative to the MFR algorithm for the soft X-ray :term:`detectors <detector>` at ASDEX Upgrade [L1]_. However, they had already been described earlier, for example in [L3]_.

The name combines the two main features of these methods. The first is the use of a linear :term:`regularisation matrix` and the second is the use of algebraic decomposition techniques for solving the tomography problem. The decomposition can also be combined with nonlinear regularisation and the iterative approach of :doc:`mfr`. Although the decomposition then has to be recomputed in each MFR iteration, as the regularisation matrix changes, the :term:`regularisation parameter` search within each iteration requires only matrix multiplications, which speeds up the inner loop.

A linear regularisation matrix does not depend on an intermediate solution, which is a simplification compared to the nonlinear regularisation :eq:`mfr_regularisation` used by MFR. This renders the iterative approach unnecessary and the solution can be obtained in a single step. The regularisation matrix consists of discretised derivative matrices :math:`\mathbf{B}_{i}` that can be weighted by coefficients :math:`c_i` to introduce anisotropy

.. math::
    \mathbf{H} = \sum\limits_i c_i \mathbf{B}^{T}_{i} \cdot \mathbf{B}_{i} \,,

where the coefficients are normalised so that :math:`\sum_i c_i = 1` when the matrix is constructed by :func:`~tomotok.regularisations.weighted_squares`. Various numerical schemes can be used for the derivative matrices and combined in the regularisation matrix.

The decomposition serves mainly for solving the inversion problem. Additionally, the regularisation parameter can be estimated from the properties of the decomposed matrices. The main advantage of this approach is that the decomposition has to be computed only once for a given geometry and regularisation matrix. The inversion of individual :term:`time slices <time slice>`, or with different values of the regularisation parameter, then requires only matrix multiplications.

Series expansion
----------------

All variants of these methods can be formulated so that they share the same baseline given by the :ref:`Phillips-Tikhonov regularisation <tikhonov>` scheme :eq:`tikhonov` [L1]_. They differ in the decomposition method used and in the method for determining the regularisation parameter. The solution of :eq:`tikhonov` is expressed as a series expansion using the decomposition matrices :math:`\mathbf{U} \in \mathbb{R}^{N_\mathrm{c}, K}`, :math:`\mathbf{D} \in \mathbb{R}^{K, K}` and :math:`\tilde{\mathbf{V}} \in \mathbb{R}^{N_\mathrm{n}, K}`

.. math::
    :label: series

    \mathbf{g}(\alpha) = \sum_{i=1}^{K} \frac{k_{i} (\alpha)}{D_{ii}} \left( \mathbf{u}_{i} \cdot \mathbf{f} \right) \tilde{\mathbf{v}}_{i} \,,

where :math:`\mathbf{u}_{i}` and :math:`\tilde{\mathbf{v}}_{i}` are the i-th columns of :math:`\mathbf{U}` and :math:`\tilde{\mathbf{V}}`, :math:`\mathbf{D}` is a diagonal matrix, :math:`K \leq N_\mathrm{c}` is the number of components used and :math:`k_{i}(\alpha)` are filter factors dependent on the regularisation parameter

.. math::
    :label: filters

    k_{i}(\alpha) = \left(1 + \frac{\alpha}{D_{ii}^2} \right)^{-1} \,,

The diagonal of :math:`\mathbf{D}` holds the (generalised) singular values of the problem. The regularisation parameter has the same meaning as in :eq:`tikhonov`, so its values can be exchanged with :class:`~tomotok.inversions.Tikhonov`. The form of the matrices :math:`\mathbf{U}` and :math:`\tilde{\mathbf{V}}` is determined by the decomposition method.

Since the components are sorted by decreasing :math:`D_{ii}`, the series can be truncated to use only the most significant components, which further increases the speed at the expense of accuracy. Notice also that the solution can be quickly recomputed for different values of the regularisation parameter, which can speed up more complex searching schemes such as the :ref:`Pearson test <pearson>`.

The two decomposition methods implemented in Tomotok are described below, focusing on the steps necessary for their implementation. A more detailed explanation is provided in the referenced articles.

Singular Value Decomposition
----------------------------

This method is based on [L1]_. As the name suggests, it uses singular value decomposition (SVD) to obtain the matrices used in equation :eq:`series`. The first step is to compute the Cholesky decomposition of the regularisation matrix :math:`\mathbf{H}`

.. math::
    \mathbf{H} = \mathbf{L} \cdot \mathbf{L}^T \,.

The obtained lower triangular matrix :math:`\mathbf{L}` is used together with the :term:`geometry matrix` to define the matrix

.. math::
    \mathbf{A} = \mathbf{L}^{-1} \cdot \mathbf{T}^T \,,

which transforms the problem into the standard form of Tikhonov regularisation with an identity regularisation matrix. The next step is the singular value decomposition of matrix :math:`\mathbf{A}^T`

.. math::
    \mathbf{A}^T = \mathbf{U} \cdot \mathbf{D} \cdot \mathbf{V}^T \,,

where :math:`\mathbf{D}` contains the singular values. The matrix :math:`\mathbf{V}` is then transformed back to the reconstruction space for use in the series expansion :eq:`series`

.. math::
    \tilde{\mathbf{V}} = {\mathbf{L}^{-1}}^{T} \cdot \mathbf{V} \,.


Generalised Eigenvalue Decomposition
------------------------------------

This method is an alternative to the SVD method described above. The computation of the matrices used in equation :eq:`series` is based on the generalised eigenvalue decomposition (GEV) of the symmetrised geometry matrix and the regularisation matrix. Below, an implementation based on [L2]_ is described. First, the symmetric matrix :math:`\mathbf{C}` is defined as

.. math::
    \mathbf{C} = \mathbf{T}^T \cdot \mathbf{T} \,.

The matrices used in equation :eq:`series` are determined by solving the generalised eigenvalue problem

.. math::
    \mathbf{C} \cdot \mathbf{V} = \mathbf{H} \cdot \mathbf{V} \cdot \mathbf{\Lambda} \,,

where the matrix :math:`\mathbf{V}` contains eigenvectors in columns, normalised so that :math:`\mathbf{V}^T \cdot \mathbf{H} \cdot \mathbf{V} = \mathbf{I}`, and :math:`\mathbf{\Lambda}` is a diagonal matrix of eigenvalues. The eigenvalues are squares of the generalised singular values, i.e. the same values that are obtained by the SVD method. Attention must be paid to the sorting of eigenvalues, as the described algorithm expects the eigenvalues and corresponding eigenvectors to be sorted from largest to smallest. This is explicitly stated because some numerical libraries return eigenvalues sorted from smallest to largest.

The remaining matrices for use in the series expansion :eq:`series` are then calculated as

.. math::
    \mathbf{D} & = \mathbf{\Lambda}^{\frac{1}{2}} \,, \\
    \mathbf{U} & = \mathbf{T} \cdot \mathbf{V} \cdot \mathbf{D}^{-1} \,, \\
    \tilde{\mathbf{V}} & = \mathbf{V} \,,

where the exponents of the diagonal matrices are applied element-wise. Substituting these into :eq:`series` gives the solution of :eq:`tikhonov` expressed in the eigenvectors, which diagonalise both :math:`\mathbf{C}` and :math:`\mathbf{H}`

.. math::
    \mathbf{g}(\alpha) = \sum_{i=1}^{K} \frac{\tilde{\mathbf{v}}_i \cdot \mathbf{T}^T \cdot \mathbf{f}}{\Lambda_{ii} + \alpha} \tilde{\mathbf{v}}_i \,.

As with the SVD method, which requires the Cholesky decomposition of :math:`\mathbf{H}`, the regularisation matrix must be positive definite. With the default ``compensate_edges=True``, every row of the derivative matrices sums to zero, so a constant :term:`emissivity` is not penalised and :math:`\mathbf{H}` is singular, with the constant vector as its null space (see :doc:`../tomography/regularisation`). This does not affect :class:`~tomotok.inversions.Tikhonov`, where the data term penalises the constant, but the algebraic methods require :math:`\mathbf{H}` itself to be positive definite. A small multiple of the identity matrix should therefore be added to :math:`\mathbf{H}`, or ``compensate_edges=False`` used. :class:`~tomotok.inversions.GevAlgebraic` does not detect a singular :math:`\mathbf{H}` and returns an incorrect result.

Implementation
--------------

The structure of the classes is based on the algorithms proposed by T. Odstrcil in [L1]_. A base class :class:`~tomotok.inversions.lame.Algebraic` implements the series expansion :eq:`series` in its ``invert`` method, while subclasses implement the ``decompose`` method. Unlike other :class:`~tomotok.inversions.base.RegularisedInversion` subclasses, algebraic methods do not use a :class:`~tomotok.inversions.solvers.Solver`. In the current implementation, the decomposition is recomputed on every call, since the inputs are normalised by the errors of the given time slice. Within a call, different values of the regularisation parameter are evaluated without recomputing the decomposition, which speeds up the search for the regularisation parameter compared to :class:`~tomotok.inversions.Tikhonov`. The ``num`` parameter limits the number of components :math:`K` used in the series expansion, and it can be changed for a single evaluation by the ``num`` argument of ``invert``. :class:`~tomotok.inversions.SvdAlgebraic` always computes all components and ``num`` only truncates the series, while :class:`~tomotok.inversions.GevAlgebraic` computes only the :math:`K` largest eigenvalues, see below.

- :class:`~tomotok.inversions.SvdAlgebraic` implements the SVD method. It works with dense matrices; sparse inputs are converted to dense format.
- :class:`~tomotok.inversions.GevAlgebraic` implements the GEV method using the sparse eigenvalue solver :func:`scipy.sparse.linalg.eigsh`. Only :math:`K` largest eigenvalues are computed (:math:`N_\mathrm{c}` by default), which makes it suitable for large :term:`grids <grid>`. If the requested number of components is not smaller than :math:`N_\mathrm{n}`, the dense solver :func:`scipy.linalg.eigh` is used instead. Components with numerically zero eigenvalues, which occur for a rank-deficient geometry matrix, are discarded.

The regularisation parameter can be determined by any :class:`~tomotok.inversions.base.RegularisationSelector`. Additionally, :class:`~tomotok.inversions.FastSelector` provides a fast estimate of :math:`\alpha` from the diagonal values of :math:`\mathbf{D}` without any additional inversion. The estimate is computed from all singular values stored by the decomposition, regardless of the truncation of the series. These are all :math:`N_\mathrm{c}` values for :class:`~tomotok.inversions.SvdAlgebraic`, but only the :math:`K` computed values for :class:`~tomotok.inversions.GevAlgebraic`, so the same ``num`` can lead to different estimates for the two methods. Supported estimates are the ``mean``, ``median``, ``quantile`` (the :math:`2/e` quantile, default), ``logmean`` (geometric mean) and ``half`` (half of the maximum value) of the diagonal values. Since the singular values enter the filter factors :eq:`filters` squared, the estimate is squared by default, which can be switched off by the ``square`` parameter.

References
----------

.. [L1] T. Odstrcil et al., "Optimized tomography methods for plasma emissivity reconstruction at the ASDEX Upgrade tokamak," Rev. Sci. Instrum., 87(12), 123505.

.. [L2] L.C. Ingesson, "The Mathematics of Some Tomography Algorithms Used at JET," JET Joint Undertaking (2000)

.. [L3] N. Terasaki et al., "Linear algebraic algorithms for high speed and stable reconstruction of plasma image," Fusion Eng. Des. 34 (1997): 801-804.
