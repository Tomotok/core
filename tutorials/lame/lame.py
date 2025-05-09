import matplotlib.pyplot as plt
import numpy as np

from tomotok.core.phantoms import gauss_iso
from tomotok.core.geometry import generate_los, sparse_line, RegularGrid
from tomotok.core.derivative import derivative_matrix, laplace_matrix
from tomotok.core.inversions import FastGevAlgebraic, FastSvdAlgebraic
from tomotok.tools.regularisation import regularisation_matrix


grid = RegularGrid(50, 100, (.2, .7), (-.5, .5))

phantom = gauss_iso(grid.nr, grid.nz, cen=.3, amp=100)

plt.figure()
plt.imshow(phantom, extent=grid.extent, origin='lower')
plt.colorbar(label='Emissivity [-]')
plt.xlabel('R [-]')
plt.ylabel('z [-]')

num = 20
s1, e1 = generate_los(num=(num, 1), fov=(70, 0), pinhole=(1, 0, 0), axis=(-1, 0, 0))
s2, e2 = generate_los(num=(num, 1), fov=(35, 0), pinhole=(.4, 0, .8), axis=(.1, 0, -1), length=1.5)
s3, e3 = generate_los(num=(num, 1), fov=(50, 0), pinhole=(.75, 0, -.4), axis=(-1, 0, 1), length=1.5)
s4, e4 = generate_los(num=(num, 1), fov=(50, 0), pinhole=(.75, 0, .4), axis=(-1, 0, -1), length=1.5)
s = np.concatenate((s1, s2, s3, s4))
e = np.concatenate((e1, e2, e3, e4))

gmat = sparse_line(s, e, grid, rmin=.2)

plt.figure()
plt.imshow(gmat.sum(0).reshape(grid.shape), extent=grid.extent, origin='lower')
plt.colorbar(label='Length [-]')
plt.xlabel('R [-]')
plt.ylabel('z [-]')

np.random.seed(20250506)
forward = gmat @ phantom.flatten()
noise = 0.05 * forward.max()
signal = forward + np.random.normal(0, noise, forward.shape)
errors = (signal + signal.max() ) / 2 * .05

plt.figure()
plt.errorbar(np.arange(signal.size), signal, yerr=errors, fmt='+', capsize=3, label='Signal with noise estimate')
plt.plot(forward, label='Exact', marker='x', ls='')
plt.xlabel('Channel [-]')
plt.ylabel('Signal [-]')
plt.legend()


svd = FastSvdAlgebraic()  # no sparse optimization

derivatives = [
    derivative_matrix(grid, 'left', compensate_edges=False),
    derivative_matrix(grid, 'bottom', compensate_edges=False),
    derivative_matrix(grid, 'right', compensate_edges=False),
    derivative_matrix(grid, 'top', compensate_edges=False),
]
# derivatives = [laplace_matrix(grid, compensate_edges=False)]

regularisation = regularisation_matrix(derivatives)

out_svd, stats_svd = svd(signal, gmat, regularisation, errors, method='median')

plt.figure()
plt.title('SVD')
plt.imshow(out_svd.reshape(grid.shape), origin='lower', extent=grid.extent)
plt.colorbar(label='Emissivity [-]')
plt.xlabel('R [-]')
plt.ylabel('z [-]')

tmp = svd.series_expansion(svd.alpha, signal/errors)
# tmp = svd.series_expansion(1, data_nrm[0])

plt.figure()
plt.plot(signal)
plt.plot(gmat @ tmp)

gev = FastGevAlgebraic()

out_gev, stats_gev = gev(signal, gmat, regularisation, errors, method='median')

plt.figure()
plt.title('GEV')
plt.imshow(out_gev.reshape(grid.shape), origin='lower', extent=grid.extent)
plt.colorbar(label='Emissivity [-]')
plt.xlabel('R [-]')
plt.ylabel('z [-]')

plt.show()
