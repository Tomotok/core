Welcome
=======
Tomotok provides tomographic inversion methods and supporting tools for tokamak plasma radiation, assuming toroidal symmetry of the emissivity.
It aims to provide multiple inversion algorithms that ease the use of tomography in the fusion community. The package is released as open source for transparency of the code, reproducibility of the results and to make it available to a broader community.

Package structure
-----------------
The inversion methods in :mod:`tomotok.inversions` form the main part of Tomotok (see :doc:`methods/methods`). These are independent of the rest of the package and accept any geometry and regularisation matrices. The other three subpackages complement them by preparing their inputs and supporting the analysis:

- :mod:`tomotok.geometry`: definition of reconstruction grids, generation of lines of sight and computation of geometry matrices.
- :mod:`tomotok.regularisations`: derivative matrices and their combination into regularisation matrices.
- :mod:`tomotok.tools`: templates of database interfaces, phantom emissivity generators and other utilities.

The complementary subpackages focus on the simplest and most common case of toroidally symmetric nodes on a regular rectangular grid; Tomotok does not aim to provide a complete synthetic diagnostic framework.

Contributors
------------

.. list-table::
    :header-rows: 1

    * - Name
      - Role
      - Scope
      - Years
    * - Jakub Svoboda
      - Lead Developer
      - core
      - 2018 -
    * - Jordan Cavalier
      - Consultant
      - biorthogonal basis decomposition
      - 2020 -
    * - Ondřej Ficker
      - Contributor
      - MFR
      - 2018 -
    * - Martin Imríšek
      - Consultant
      - MFR
      - 2018 -
    * - Jan Mlynář
      - Mentoring
      - \-
      - 2018 - 2023

Citing the code
---------------
If used for a scientific publication, please cite the software using the reference below together with the references for the inversion methods used. These can be found in :doc:`methods/methods`.

"J. Svoboda, J. Cavalier, O. Ficker, M. Imríšek, J. Mlynář and M. Hron, Tomotok: python package for tomography of tokamak plasma radiation, Journal of Instrumentation 16.12 (2021): C12015." `DOI 10.1088/1748-0221/16/12/C12015 <https://doi.org/10.1088/1748-0221/16/12/C12015>`_

.. code-block:: bibtex

    @article{svoboda2021tomotok,
        author = {Svoboda, J. and Cavalier, J. and Ficker, O. and Imríšek, M. and Mlynář, J. and Hron, M.},
        title = {Tomotok: python package for tomography of tokamak plasma radiation},
        journal = {Journal of Instrumentation},
        volume = {16},
        number = {12},
        pages = {C12015},
        year = {2021},
        doi = {10.1088/1748-0221/16/12/C12015},
    }
