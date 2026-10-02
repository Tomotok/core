Inversions
==========

Inversion methods are independent of the rest of the package and can be used on any data if the required inputs are provided. However, the rest of this package provides these inputs only under the assumption of toroidal symmetry and for regularly spaced reconstruction nodes with rectangular cross-section in the reconstruction plane. This was selected as it is the simplest and most widely used option for reconstructions of tokamak plasma. For a more detailed description, see for example [I1]_, which is a comprehensive overview article on tomography of tokamak plasma.

.. note::
    The descriptions of the algorithms in this chapter draw on the overview of tomographic algorithms in section 1.2 of [I2]_. Where the formulation in the thesis and the implementation differ, the documentation follows the implementation.

Tomography problem
------------------

Tomographic inversion uses line-integrated (or volume-integrated) measurements to determine the spatial distribution of a physical quantity. In the case of tokamak plasma, this quantity is usually electromagnetic radiation emitted by the plasma. Various parts of the emission spectrum are used for inversion, depending on the detector and the physics to be studied. This radiation can be recorded by metallic foil bolometers, photodiode arrays or even fast visible cameras. All of these systems consist of multiple sensitive elements, here referred to as channels.

The emissivity is expanded into a series of basis functions. All methods implemented here use local basis functions (disjoint compact basis functions that are equal to one in a given node and zero elsewhere), which is why they are also referred to as discretisation or series expansion methods. The main advantage of local basis functions is their universality and simple implementation. With a sufficiently large number of nodes, no prior knowledge about the plasma shape or position of the magnetic axis is required and a regular grid can be used. Information about the plasma shape can instead be introduced by the inversion algorithm itself, e.g. by anisotropic regularisation.

With this discretisation, the tomography problem can be written as

.. _tomobase:

.. math::
    \mathbf{f} = \mathbf{T} \cdot \mathbf{g} \,,

where :math:`\mathbf{f} \in \mathbb{R}^{N_\mathrm{c}}` is the measured signal of :math:`N_\mathrm{c}` channels, :math:`\mathbf{g} \in \mathbb{R}^{N_\mathrm{n}}` is the discretised emissivity in :math:`N_\mathrm{n}` nodes of the reconstruction plane and :math:`\mathbf{T} \in \mathbb{R}^{N_\mathrm{c}, N_\mathrm{n}}` is the geometry matrix describing the spatial sensitivity of the detectors (also called contribution or sensitivity matrix). The product :math:`\mathbf{T} \cdot \mathbf{g}` is called the retrofit or forward model; it represents the signal that would be recorded for emissivity :math:`\mathbf{g}`.

Under- and over-determined problems
-----------------------------------

Tomography is typically used with data from linear systems, i.e. systems consisting of individual detectors placed next to each other on a line or small detector arrays creating a fan-like configuration of lines of sight. These usually have significantly fewer channels than nodes used for the inversion (:math:`N_\mathrm{c} < N_\mathrm{n}`), which makes the problem under-determined with an infinite number of possible solutions. A priori information has to be added to select a unique solution. This is usually done with a regularisation matrix, which commonly consists of discretised derivatives and penalises solutions that are not smooth. Such methods are described below and implemented in :doc:`mfr` and :doc:`lame`.

Systems based on cameras or other matrix detectors can have more pixels than there are nodes (:math:`N_\mathrm{c} \geq N_\mathrm{n}`), making the problem over-determined. These systems do not necessarily require regularisation to obtain a unique solution, and unregularised methods such as :doc:`bob` can be used. Regularisation can still improve the solution if the camera view does not cover the vessel well.

.. _tikhonov:

Phillips-Tikhonov regularisation
--------------------------------

Phillips-Tikhonov regularisation is a mathematical recipe for solving ill-posed problems. It adds a term to the problem that prefers solutions with certain properties. Applying it to the :ref:`tomography problem <tomobase>` results in [I1]_

.. math::
   :label: tikhonov

   ( \mathbf{T}^T \cdot \mathbf{T} + \alpha \mathbf{H} ) \cdot \mathbf{g} = \mathbf{T}^T \cdot \mathbf{f} \,,

where :math:`\mathbf{H} \in \mathbb{R}^{N_\mathrm{n}, N_\mathrm{n}}` is the regularisation matrix and :math:`\alpha` is the regularisation parameter determining the strength of the a priori information relative to the measured data. The standard Tikhonov regularisation uses the identity matrix, penalising solutions with a large norm. This is not suitable for plasma radiation tomography and regularisation matrices constructed from discretised derivatives are used instead, since most physical properties of the plasma are expected to change smoothly. The signals and the geometry matrix are normalised by the expected errors of individual channels before the inversion, so that channels with lower uncertainty have larger weight.

In Tomotok, equation :eq:`tikhonov` is solved by :class:`~tomotok.inversions.tikhonov.Tikhonov`. The linear system is solved by an interchangeable :class:`~tomotok.inversions.solvers.Solver` backend (dense or sparse Cholesky decomposition, non-negative least squares and others, see :mod:`tomotok.inversions.solvers`).

.. _pearson:

Regularisation parameter
^^^^^^^^^^^^^^^^^^^^^^^^

There are multiple methods for determining the regularisation parameter, see [I1]_ or [L1]_. A method that has proved reliable is based on the Pearson :math:`\chi^2` test. It determines the optimal value of :math:`\alpha` by comparing the misfit of the reconstruction with the estimated error of the measured data

.. math::
   \chi^2 = \frac{1}{N_\mathrm{c}} \sum_{i=1}^{N_\mathrm{c}} \left( \frac{ f_i - \left( \mathbf{T} \cdot \mathbf{g} \right)_i }{\sigma_i} \right)^2 \,,

where :math:`f_i` is the signal measured by the i-th channel and :math:`\sigma_i` is its expected error. The optimal value is considered to be one [M2]_, which means that the misfit of the reconstruction is approximately the same as the estimated error of the measured data.

This test is implemented in :class:`~tomotok.inversions.base.PearsonSelector`, which is the default regularisation parameter selector. It minimises :math:`|\chi^2 - 1|` using a bounded search algorithm combined with Brent's method (:func:`scipy.optimize.minimize_scalar`) [I3]_ [I4]_. The search is conducted on a logarithmic scale within bounds representing values of :math:`\alpha` that are considered too small or too large, by default :math:`10^{-30} \leq \alpha \leq 10^{10}`. The main disadvantage of this approach is its computational cost, as one inversion of :eq:`tikhonov` is required for each tested value of :math:`\alpha`.

Repeated optimisation can be avoided by using :class:`~tomotok.inversions.base.FixedSelector` with a regularisation parameter provided by the user. This utilises the fact that optimisation on similar data (e.g. consecutive time slices) provides similar values of the regularisation parameter. Linear algebraic methods additionally provide a fast estimate based on their decomposition, see :doc:`lame`.

Implemented inversion methods
-----------------------------

.. toctree::
    :maxdepth: 1

    mfr
    lame
    bob

.. rubric:: References

.. [I1] L. Ingesson, B. Alper, B. Peterson and J.-C. Vallet, Chapter 7: Tomography diagnostics: Bolometry and soft-x-ray detection, Fusion Sci. Technol. 53 (2008) 528.

.. [I2] J. Svoboda, Design and applications of tomography diagnostics for tokamaks, Doctoral thesis, Czech Technical University in Prague (2025), http://hdl.handle.net/10467/126851

.. [I3] G. E. Forsythe, M. A. Malcolm and C. B. Moler, Computer Methods for Mathematical Computations, Prentice-Hall (1977).

.. [I4] R. P. Brent, Algorithms for Minimization without Derivatives, Prentice-Hall (1973).
