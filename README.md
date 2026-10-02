# Tomotok
Tomotok is a framework for the tomographic inversion of fusion plasmas, focusing on inversion methods based on discretisation. It is structured as a namespace package to ease implementation on different experimental devices and various diagnostics.

Documentation built from the `development` branch of this repository, tracking the latest changes, is available at https://tomotok.github.io/core/.

The core package of the Tomotok namespace implements various discretisation algorithms that are used for tomographic reconstruction of tokamak plasmas. It is a dependency for specific packages that automate database access for a given fusion experimental device.

## Installation
The package can be installed using pip:
```bash
pip install tomotok
```
The PyPI packages might not contain the most recent features. 
It is recommended to install the package from source, if you want to use the latest features and do not mind the possibility of backward compatibility issues.

### Source code
The source code can be found in the repository on [GitHub](https://github.com/Tomotok/core).
The `stable` branch ensures backward compatibility within the latest major version.
Newest features with not fully guaranteed backward compatibility are implemented in the `development` branch. Tags follow the versions published on [PyPI](https://pypi.org/project/tomotok/).

## Contents

The inversion methods in `tomotok.inversions` form the main part of Tomotok. They accept inputs in the form of `numpy.ndarray` or `scipy.sparse` objects, allowing them to run independently of the rest of the package. There are three main types of algorithms implemented:
- Minimum Fisher Regularisation (MFR)
- Linear algebraic methods (LAME)
- Biorthogonal Basis Decomposition (BoB)

The other three subpackages complement the inversion methods by preparing their inputs and supporting the analysis:
- `tomotok.geometry`: reconstruction grids, generation of lines of sight and geometry matrix computation using numerical integration and a single line of sight approximation
- `tomotok.regularisations`: derivative matrices, both isotropic and anisotropic (based on magnetic flux surfaces), and their combination into regularisation matrices
- `tomotok.tools`: templates of database interfaces, phantom emissivity generators and other utilities

The complementary subpackages assume toroidal symmetry and regular rectangular nodes, as it is the simplest case often used for inversions of tokamak plasma radiation.

## Citing the Code

When used for research purposes, please cite the following paper. To cite specific inversion methods, please refer to the corresponding papers included in the documentation.

"J. Svoboda, J. Cavalier, O. Ficker, M. Imríšek, J. Mlynář and M. Hron, *Tomotok: python package for tomography of tokamak plasma radiation*, Journal of Instrumentation 16.12 (2021): C12015."
[DOI 10.1088/1748-0221/16/12/c12015](https://doi.org/10.1088/1748-0221/16/12/c12015)

## Contributors

| Name | Role | Scope | Years |
|---|---|---|---|
| Jakub Svoboda (@skuba31) | Lead Developer | core | 2018 - |
| Jordan Cavalier | Consultant | biorthogonal basis decomposition | 2020 - |
| Ondřej Ficker | Contributor | MFR | 2018 - |
| Martin Imríšek | Consultant | MFR | 2018 - |
| Jan Mlynář | Mentoring | - | 2018 - 2023 |
