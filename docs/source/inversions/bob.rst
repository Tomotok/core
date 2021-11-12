BiOrthogonal Basis Decomposition
================================

This method is based on a wavelet-vaguelette algorithm [B1]_ and on a modified version that was used for plasma edge observation and described in [B2]_ . 

This method is unregularised and therefore it requires the number of channels to be larger (or equal) than the number of nodes (see chapter 5 of [B3]_ ) i.e. the task  has to be over determined. It is therefore not suitable for systems consisting of linear detectors where the number of channels is usually significantly lower, but it suits well matrix camera based systems.

The main advantage of this approach is that once the decomposition matrix is computed it stays the same for given reconstruction grid and geometry setup. The results are then obtained simply by using matrix multiplication.

Another convenient feature is thresholding. This methods selects only the contributions with the greatest norm to isolate only the most prominent features in the reconstruction plane.

The main principle relies on using a biorthogonal dual basis set in the image space. Using the properties of adjugated matrices and biorthogonal transform a relation between the observed image and the reconstruction space can be determined.
Assuming that the adjugated matrix :math:`\mathbf{T}^*` of the matrix :math:`\mathbf{T}` exists, it is possible to write the following system [B1]_

.. math::
    \mathbf{T} \cdot \mathbf{b}_{i} = \kappa_{i} \mathbf{e}_{i} \,, \qquad
    \mathbf{T}^{*} \cdot \hat{\mathbf{e}}_{i} = \kappa_{i} \mathbf{b}_i \,,

where :math:`\mathbf{b}_i` are basis vectors in the reconstruction space, :math:`\mathbf{e}_i` and :math:`\hat{\mathbf{e}}_i` form a dual set of vectors in the image space built in such a way that it is biorthogonal and :math:`\kappa_{i}` are normalisation factors for the set of basis vectors :math:`\mathbf{e}_{i}` that serve for denoising purposes [B1]_ .

For any two vectors and matrix with its adjoint, following relation is valid

.. math::
	( \mathbf{T} \cdot \mathbf{x}) \cdot \mathbf{y} = \mathbf{x} \cdot (\mathbf{T}^{*} \cdot \mathbf{y} ).

Using the above relations the resulting emissivity can be expressed in the basis :math:`\mathbf{b}_{i}` as

.. math::
    \mathbf{g} = \sum_i^{N_\mathrm{n}} \left( \mathbf{b}_{i} \cdot \mathbf{g} \right) \cdot \mathbf{b}_{i}
     = \sum_i^{N_\mathrm{n}} \frac{1}{\kappa_{i}} \left( \hat{\mathbf{e}}_{i} \cdot \mathbf{f} \right) \cdot \mathbf{b}_{i} \,.

This shows that the knowledge of the basis vectors :math:`\hat{\mathbf{e}}_{i}` would allow to recover the emissivity :math:`\mathbf{g}` from the captured image :math:`\mathbf{f}`.
The basis vectors :math:`\hat{\mathbf{e}}_{i}` can be found by applying the biorthogonal condition of the dual set :math:`(\mathbf{e}_{i}, \hat{\mathbf{e}}_{i})` [B2]_. Note again that it is required that :math:`N_\mathrm{n} \le N_\mathrm{c}`.

Implementation
--------------

Dense matrices only. No wavelet support implemented yet. Simple one node basis is used.

References
----------

.. [B1] R. Nguyen Van Yen, N. Fedorczak, F. Brochard, G. Bonhomme, K. Schneider, M. Farge et al., Tomographic reconstruction of tokamak plasma light emission from single image using wavelet-vaguelette decomposition, Nucl. Fusion 52 (2011) 013005.

.. [B2] J. Cavalier, N. Lemoine, F. Brochard, V. Weinzettl, J. Seidl, S. Silburn et al., Tomographic reconstruction of tokamak edge turbulence from single visible camera data and automatic turbulence structure tracking, Nucl. Fusion 59 (2019) 056025.

.. [B3] S. Mallat, A Wavelet Tour of Signal Processing - The Sparse Way. Elsevier, third ed., 2009.
