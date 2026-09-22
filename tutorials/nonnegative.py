from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from tomotok.geometry import sparse_line, RegularGrid, generate_sightlines
from tomotok.inversions import PearsonSelector, Tikhonov
from tomotok.inversions.solvers.scipy import NNLSSolver
from tomotok.regularisations import all_direction_derivative_matrices, weighted_squares

from tomotok.tools.phantoms import elliptical_flux, gaussian_on_flux

USE_CVXPY = False  # requires cvxpy, typically 2x faster than Scipy
USE_OPTAX = False  # requires jax and optax, typically 3x faster than Scipy, dense matrices only

# very fast computation, too rough for nice results, used for automatic testing
grid = RegularGrid(20, 20, (.2, .7), (-.5, .5))

# middle ground with reasonable computation time, recommended for manual inspection
# grid = RegularGrid(50, 50, (.2, .7), (-.5, .5))

# very fine grid with extensive computation time
# grid = RegularGrid(100, 100, (.2, .7), (-.5, .5))

# Artificial emissivity, particularly challenging hollow phantom
flux = elliptical_flux(grid.nr, grid.nz, span=1.3)  # artificial flux values
phantom = gaussian_on_flux(flux, center=0.3, amplitude=100, limit_width=0.2)  # flux dependent emissivity

plt.figure()
plt.imshow(phantom, extent=grid.extent, origin='lower')
plt.colorbar(label='Emissivity [-]')
plt.xlabel('R [-]')
plt.ylabel('z [-]')

# Detector setup and geometry matrix
num = 20
s1, e1 = generate_sightlines(num=(num, 1), fov=(70, 0), pinhole=(1, 0, 0), axis=(-1, 0, 0))
s2, e2 = generate_sightlines(num=(num, 1), fov=(35, 0), pinhole=(.4, 0, .8), axis=(.1, 0, -1), length=1.5)
s3, e3 = generate_sightlines(num=(num, 1), fov=(50, 0), pinhole=(.75, 0, -.4), axis=(-1, 0, 1), length=1.5)
s4, e4 = generate_sightlines(num=(num, 1), fov=(50, 0), pinhole=(.75, 0, .4), axis=(-1, 0, -1), length=1.5)
s = np.concatenate((s1, s2, s3, s4))
e = np.concatenate((e1, e2, e3, e4))

# Geometry matrix computation
gmat = sparse_line(s, e, grid, rmin=.2)

plt.figure()
plt.imshow(gmat.sum(0).reshape(grid.shape), extent=grid.extent, origin='lower')
plt.colorbar(label='Length [-]')
plt.xlabel('R [-]')
plt.ylabel('z [-]')

# Artificial signal
np.random.seed(20250506)
fwd = gmat @ phantom.flatten()
noise = 0.05 * fwd.max()
signal = fwd
# signal += np.random.normal(0, noise, fwd.shape)

plt.figure()
plt.plot(fwd, '+', label='forward')
plt.plot(signal, '+', label='noisy')
plt.xlabel('Channel [-]')
plt.ylabel('Signal [-]')
plt.legend()

errors = (signal + signal.max() ) / 2 * 0.02
# errors = 0.001  # constant error estimate for all channels

# Derivatives and regularisation matrix
dmats = all_direction_derivative_matrices(grid, compensate_edges=False)
reg = weighted_squares(dmats)

# Inversions
# Standard using cholesky decomposition
select = PearsonSelector(bounds=(-10, -4), iter_max=20, tolerance=0.001)
chol = Tikhonov(regularisation_selector=select)

out_chol, stats_chol = chol(signal, gmat, reg, errors)

pprint(stats_chol)

plt.figure()
plt.imshow(out_chol.reshape(grid.shape), extent=grid.extent, origin='lower')
plt.colorbar(label='Emissivity [-]')
plt.xlabel('R [-]')
plt.ylabel('z [-]')
plt.title('Cholesky')

plt.figure()
plt.plot(signal, label='signal')
plt.plot(gmat @ out_chol, label='retrofit')
plt.legend()
plt.xlabel('Channel [-]')
plt.ylabel('Signal [-]')
plt.title('Cholesky')

# Scipy Non-Negative Least Squares (NNLS)
# sparse support, single core execution
nnls = Tikhonov(regularisation_selector=select, solver=NNLSSolver())

out_nnls, stats_nnls = nnls(signal, gmat, reg, errors)

pprint(stats_nnls)

plt.figure()
plt.imshow(out_nnls.reshape(grid.shape), extent=grid.extent, origin='lower')
plt.colorbar(label='Emissivity [-]')
plt.xlabel('R [-]')
plt.ylabel('z [-]')
plt.title('SCIPY NNLS')

plt.figure()
plt.plot(signal, label='signal')
plt.plot(gmat @ out_nnls, label='retrofit')
plt.xlabel('Channel [-]')
plt.ylabel('Signal [-]')
plt.title('SCIPY NNLS')
plt.legend()

# Optional dependence: CVXPY
# convex optimization, default backend supports sparse matrices and multicore computation
# typically 2x faster than NNLS form scipy
if USE_CVXPY:
    from tomotok.inversions.solvers.cvxpy import CvxpyNNLS

    cvxp = Tikhonov(regularisation_selector=select, solver=CvxpyNNLS())
    out_cvxp, stats_cvxp = cvxp(signal, gmat, reg, errors)

    pprint(stats_cvxp)

    plt.figure()
    plt.imshow(out_cvxp.reshape(grid.shape), extent=grid.extent, origin='lower')
    plt.colorbar(label='Emissivity [-]')
    plt.xlabel('R [-]')
    plt.ylabel('z [-]')
    plt.title('CVXPY NNLS')

    plt.figure()
    plt.plot(signal, label='signal')
    plt.plot(gmat @ out_cvxp, label='retrofit')
    plt.xlabel('Channel [-]')
    plt.ylabel('Signal [-]')
    plt.title('CVXPY NNLS')
    plt.legend()

# Optional dependence: OPTAX
# dense matrices only, gradient-based NNLS via jax/optax
# typically 3x faster than Scipy
if USE_OPTAX:
    from tomotok.inversions.solvers.optax import OptaxNNLS

    optaxnnls = Tikhonov(regularisation_selector=select, solver=OptaxNNLS())
    out_optaxnnls, stats_optaxnnls = optaxnnls(signal, gmat, reg, errors)

    pprint(stats_optaxnnls)

    plt.figure()
    plt.imshow(out_optaxnnls.reshape(grid.shape), extent=grid.extent, origin='lower')
    plt.colorbar(label='Emissivity [-]')
    plt.xlabel('R [-]')
    plt.ylabel('z [-]')
    plt.title('OPTAX NNLS')

    plt.figure()
    plt.plot(signal, label='signal')
    plt.plot(gmat @ out_optaxnnls, label='retrofit')
    plt.xlabel('Channel [-]')
    plt.ylabel('Signal [-]')
    plt.title('OPTAX NNLS')
    plt.legend()

plt.show()
