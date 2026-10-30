Tomographic inversion
=====================

This chapter describes the general theoretical background of the tomographic inversion as implemented in Tomotok: the formulation of the tomography problem, the computation of the :term:`geometry matrix`, the regularisation and testing using :term:`phantoms <phantom>`. The individual inversion methods are described in :doc:`../methods/methods`. Each section also describes the corresponding implementation in Tomotok; the full description of classes and functions can be found in the API reference.

The inversion methods are independent of the rest of the package and can be used on any data if the required inputs are provided. The rest of the package provides these inputs only under the assumption of toroidal symmetry and for regularly spaced reconstruction :term:`nodes <node>` with rectangular cross-section in the :term:`reconstruction plane`, as it is the simplest and most widely used option for reconstructions of tokamak plasma. For a more detailed description, see for example [I1]_, which is a comprehensive overview article on tomography of tokamak plasma.

.. note::
    The descriptions in this chapter and in :doc:`../methods/methods` draw on chapter 1 of [I2]_, mainly on the overview of tomographic algorithms in section 1.2. Where the formulation in the thesis and the implementation differ, the documentation follows the implementation.

.. toctree::
    :maxdepth: 2

    problem
    geometry
    regularisation
    phantoms

.. rubric:: References

.. [I1] L. Ingesson, B. Alper, B. Peterson and J.-C. Vallet, Chapter 7: Tomography diagnostics: Bolometry and soft-x-ray detection, Fusion Sci. Technol. 53 (2008) 528.

.. [I2] J. Svoboda, Design and applications of tomography diagnostics for tokamaks, Doctoral thesis, Czech Technical University in Prague (2025), http://hdl.handle.net/10467/126851
