Linear Algebraic Methods
========================

These inversion algorithms are based on a decomposition of geometry matrix and emissivity independent regularisation matrix :math:`\mathbf{H}`. This matrix can be computed using the following formula:

.. math::
    \mathbf{H} = \mathbf{D}^{T}_\mathrm{R} \cdot \mathbf{D}_\mathrm{R} + \mathbf{D}^{T}_\mathrm{z} \cdot \mathbf{D}_\mathrm{z} \,,

where :math:`\mathbf{D}_{R/z}` is the discrete derivation matrix normalised by the distance of node centers. Various numerical schemes can be used for computation of derivation matrices. The regularisation matrix computation is simplified compared to MFR and does not include the nonlinear term. This enables estimation of regularisation parameter directly from the decomposed matrices.

All methods share the same baseline that is used by the Phillips-Tikhonov regularisation scheme. First a decomposition using SVD or GEV scheme is performed resulting in multiple matrices that can be modified to fit the pattern needed to solve the Phillips-Tikhonov scheme using series expansion as shown below. The resulting emissivity :math:`\mathbf{g}(\alpha)` is dependent on the regularisation parameter and can be computed using the following formula for reconstruction of :math:`N_\mathrm{n}` nodes and :math:`N_\mathrm{c}` channels [L1]_

.. math::
    \mathbf{g}(\alpha) = \sum_{i=1}^{N_\mathrm{c}} \frac{k_{i} (\alpha)}{S_{ii}} \left( \mathbf{U}^T \cdot \mathbf{f} \cdot \tilde{\mathbf{V}} \right) {}_{*i} \,,
    :label: series

where :math:`\mathbf{U} \in \mathbb{R}^{N_\mathrm{n},N_\mathrm{c}}`, :math:`\mathbf{S} \in \mathbb{R}^{N_\mathrm{c},N_\mathrm{c}}` is a diagonal matrix, :math:`\tilde{\mathbf{V}} \in \mathbb{R}^{N_\mathrm{c}, N_\mathrm{c}}` and :math:`k_{i}(\alpha)` are coefficients dependent on the value of the regularisation parameter computed by the following equation

.. math::
    k_{i}(\alpha) = \left(1 + \frac{\alpha}{S_{ii}^2} \right)^{-1} \,.

Singular Value Decomposition
----------------------------

Based on [L1]_. As the name suggest, this method uses singular value decomposition to obtain matrices used in equation :eq:`series`. The first step is to compute a Cholesky decomposition of derivative matrix :math:`\mathbf{H}`

.. math::
    \mathbf{H} = \mathbf{L} \cdot \mathbf{L}^T.

A lower triangular matrix :math:`\mathbf{L}` is obtained and an arbitrary matrix :math:`\mathbf{A}` is defined using it together with geometry matrix

.. math::
    \mathbf{A} = \mathbf{L}^{-1} \cdot \mathbf{T} .

The next step involves computing singular value decomposition of arbitrary matrix :math:`\mathbf{A}`.

.. math::
    \mathbf{A}^T = \mathbf{U} \cdot \mathbf{D} \cdot \mathbf{V}^T

The decomposed matrix :math:`\mathbf{V}` is then modified for use in the series expansion :eq:`series`.

.. math::
    \tilde{\mathbf{V}} = {\mathbf{L}^{-1}}^{T} \cdot \mathbf{V}


Generalised Eigenvalue Decomposition
------------------------------------

This method is an alternative to the SVD method described above. The computation of matrices to be used in equation :eq:`series` is based on generalised eigenvalue decomposition.
    
Below an implementation based on [L2]_ is described. First a symmetric geometry matrix :math:`\mathbf{C}` is defined as 

.. math::
    \mathbf{C} = \mathbf{T}^T \cdot \mathbf{T}.

This matrix is then used as the matrix for investigation in eigenvalue problem. The diagonal matrix to be used in equation (\ref{eq:lame-series}) is directly determined by solving generalised eigenvalue problem

.. math::
    \mathbf{C} \cdot \mathbf{V} = \mathbf{D} \cdot \mathbf{H} \cdot \mathbf{V} .

Here the :math:`\mathbf{V}` matrix is containing eigenvectors in columns. Attention must be paid to sorting of eigenvalues as described algorithm expects eigenvalues and appropriate eigenvectors to be sorted from largest to smallest. This is explicitly stated because some numeric libraries contain implementations that return eigenvalues sorted from smallest to largest.

The remaining matrices for use in series expansion (\ref{eq:lame-series}) are then calculated as

.. math::
    \tilde{\mathbf{V}} & = \mathbf{T} \cdot \mathbf{V} \cdot \mathbf{D}^{-\frac{1}{2}} \\
    \mathbf{U} & = \mathbf{D}^{\frac{1}{2}} \cdot \mathbf{V} .

Exponent of matrix :math:`\mathbf{D}` is used element wise.

Implementation
--------------

Structure of classes is based on algorithms proposed by T. Odstrcil in [L1]_. However, they are currently implemented only for dense matrices without sparse optimization. If properly implemented using sparse matrices this class of algorithms can provide fast results with reasonable accuracy.

Several decomposition schemes can be used, therefore, a base class for these algorithms called Algebraic was created. The regularisation parameter can be found using a variety of methods that can be simple or more complex. Currently. only fast estimation based on decompositions is implemented in a subclass FastAlgebraic.

Singular value decomposition (SVD) and generalised eigenvalue decomposition (GEV) methods are subclasses of FastAlgebraic.

References
----------

.. [L1] T. Odstrcil et al., "Optimized tomography methods for plasma emissivity reconstruction at the ASDEX Upgrade tokamak," Rev. Sci. Instrum., 87(12), 123505.

.. [L2] L.C. Ingesson, "The Mathematics of Some Tomography Algorithms Used at JET," JET Joint Undertaking (2000)
