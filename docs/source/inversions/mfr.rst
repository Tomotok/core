Minimum Fisher Regularization
=============================

Minimum Fisher information Regularisation algorithm based on [M1]_ and further improvements described in [M2]_ and building on other work such as [M3]_.

This algorithm relies on Tikhonov regularisation to compensate under-determined nature of the tomography problem and consist of two loops. The inner loop searches for the optimal value of regularisation parameter based on the intermediate result and signal. The outer loop regularises the problem using the optimal values of regularisation parameter found by the inner loop that determines the strength of a-priori information imposed by smoothing matrix. It is based on the intermediate result and therefore the method is nonlinear. 
	
The inner loop searches for the regularisation parameter :math:`\alpha` iteratively. It is tested against the quality of reconstruction, that is determined by Pearson :math:`\chi^2` test with each iteration. The residuum :math:`\chi^2` is determined by the following equation:

.. math::
   \chi^2 = \frac{1}{N_\mathrm{c}} \sum_{i}^{N_\mathrm{c}} \left( \frac{ \left( \mathbf{f} - \mathbf{T} \cdot \mathbf{g} \right)_i }{\mathbf{\sigma}_i} \right)^2 \,,

where :math:`N_\mathrm{c}` is the number of channels and :math:`\mathbf{\sigma}_i` is estimated noise measured by i-th channel. The optimal value is considered to be one. In such case the error of reconstruction is approximately the same as estimated error of measured data.
	
The outer loop uses following equation for inversion. It can be obtained by applying Tikhonov regularisation to tomography problem :ref:`base equation <tomobase>` [L2]_:

.. math::
   ( \mathbf{T}^T \cdot \mathbf{T} + \alpha \mathbf{H} ) \cdot \mathbf{g} = \mathbf{T}^T \cdot \mathbf{f} \,,
   :label: tichonov

where :math:`\alpha` is a regularisation parameter and :math:`\mathbf{H}` is regularisation matrix computed as:

.. math::
   \mathbf{H} = c_1 \mathbf{D}^{T}_{1} \cdot \mathbf{w} \cdot \mathbf{D}_{1} + c_2 \mathbf{D}^{T}_{2} \cdot \mathbf{w} \cdot \mathbf{D}_{2} \,,

where :math:`\mathbf{D_1}`, :math:`\mathbf{D_2}` are matrices of numerical derivatives along locally orthogonal directions, :math:`c_1`, :math:`c_2`
anisotropy coefficients and :math:`\mathbf{w}` is a matrix with weights for individual nodes. Elements of the weight matrix are computed as :math:`1/\mathbf{g}_0` for non zero elements of :math:`\mathbf{g}_0`, otherwise, a predefined value :math:`w_{\mathrm{max}}` is used. The sum of anisotropy coefficients should be equal to 1. The value of the regularisation parameter is tested using Pearson :math:`\chi^2` test. The optimal value of the regularisation parameter is then used to update the regularisation matrix and the inner loops repeats until the maximum number of outer cycles is reached.

Implementation
--------------

There are two version of the MFR algorithm currently implemented. One relies on scipy.sparse library which is more commonly available but it is slower. The other uses cholesky decomposition cholmod from scikit sparse library for solving the regularised problem :eq:`tichonov`. This method is significantly faster for large grids.

References
----------

.. [M1] M. Anton et al., "X-ray tomography on the TCV tokamak.", Plasma Phys. Control. Fusion  38.11 (1996): 1849.

.. [M2] J. Mlynar et al., "Current research into applications of tomography for fusion diagnostics." J. Fusion Energy 38.3 (2019): 458-466.

.. [M3] M. Imrisek et al., "Optimization of soft X-ray tomography on the COMPASS tokamak." Nukleonika 61, (2016).
