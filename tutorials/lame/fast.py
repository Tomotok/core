import matplotlib.pyplot as plt
import numpy as np

from tomotok.geometry import sparse_line, RegularGrid, generate_sightlines
from tomotok.inversions.lame import GevAlgebraic, SvdAlgebraic, FastSelector
from tomotok.regularisations import derivative_matrix, weighted_squares
from tomotok.tools.phantoms import gaussian_on_flux, elliptical_flux


# Setup
# -----
# grid definition
grid = RegularGrid(50, 100, (.2, .7), (-.5, .5))
# artificial emissivity distribution
flux = elliptical_flux(grid.nr, grid.nz)
phantom = gaussian_on_flux(flux, center=0.3, amplitude=100) 

# definition of detector placement and geometry
num = 20
s1, e1 = generate_sightlines(num=num, fov=70, pinhole=(1.0, 0, 0), axis=(-1, 0, 0))
s2, e2 = generate_sightlines(num=num, fov=35, pinhole=(0.4, 0, 0.8), axis=(0.1, 0, -1), length=1.5)
s3, e3 = generate_sightlines(num=num, fov=50, pinhole=(0.75, 0, -0.4), axis=(-1, 0, 1), length=1.5)
s4, e4 = generate_sightlines(num=num, fov=50, pinhole=(0.75, 0, 0.4), axis=(-1, 0, -1), length=1.5)
startpoints = np.concatenate((s1, s2, s3, s4))
endpoints = np.concatenate((e1, e2, e3, e4))

# geometry matrix computation using line of sight approximation
gmat = sparse_line(startpoints, endpoints, grid, rmin=.2)

# generation of artificial signal with added noise
forward = gmat @ phantom.flatten()
rng = np.random.default_rng(20250506)
noise = 0.05 * forward.max()
signal = forward + rng.normal(0, noise, forward.shape)
errors = (signal + signal.max() ) / 2 * .05  # noise level estimates used in inversion

# regularisation matrix computation using first order derivatives in four directions
derivatives = [
    derivative_matrix(grid, 'left', compensate_edges=False),
    derivative_matrix(grid, 'bottom', compensate_edges=False),
    derivative_matrix(grid, 'right', compensate_edges=False),
    derivative_matrix(grid, 'top', compensate_edges=False),
]
regularisation = weighted_squares(derivatives)


# Inversions
# ----------
# SVD inversion, currently without sparse optimisation
svd = SvdAlgebraic(regularisation_selector=FastSelector(method='median'))
out_svd, stats_svd = svd(signal, gmat, regularisation, errors)
# Compute inversion for a specified regularisation parameter without need to repeat the decomposition
tmp = svd.invert(stats_svd['alpha'])
retrofit = gmat @ tmp

# GEV inversion
gev = GevAlgebraic(regularisation_selector=FastSelector(method='median'))
out_gev, stats_gev = gev(signal, gmat, regularisation, errors)
alpha_median = stats_gev['alpha']
# additional methods for fast regularisation parameter selection
gev.selector.method = 'quantile'
alpha_quant, stats_quant = gev.determine_regularisation()
out_gev_quantile = gev.invert(alpha_quant)
gev.selector.method = 'logmean'
alpha_logm, stats_logm = gev.determine_regularisation()
out_gev_logm = gev.invert(alpha_logm)

# Figures
# -------
# phantom and geometry
fig_inp = plt.figure()
gs_inp = fig_inp.add_gridspec(1, 4, width_ratios=[1, 0.05]*2, wspace=0.2)
ax1 = fig_inp.add_subplot(gs_inp[0, 0])
ax1.set_title('Phantom')
im_ph = ax1.imshow(phantom, extent=grid.extent, origin='lower')
ax1.set_xlabel('R [-]')
ax1.set_ylabel('z [-]')

cax1 = ax1.inset_axes([1.05, 0, 0.05, 1])
fig_inp.colorbar(im_ph, cax=cax1, label='Emissivity [-]')

ax2 = fig_inp.add_subplot(gs_inp[0, 2])
ax2.label_outer()
ax2.set_title('Geometry')
im_gmat = ax2.imshow(gmat.sum(0).reshape(grid.shape), extent=grid.extent, origin='lower')
ax2.set_xlabel('R [-]')

cax2 = ax2.inset_axes([1.05, 0, 0.05, 1])
fig_inp.colorbar(im_gmat, cax=cax2, label='Length [-]')


# exact forward model and signal with added noise and noise level estimates
fig_signal = plt.figure()
ax = fig_signal.add_subplot(1, 1, 1)
ax.errorbar(np.arange(signal.size), signal, yerr=errors, fmt='+', capsize=3, label='Signal with noise estimate')
ax.plot(forward, label='Exact', marker='x', ls='')
ax.set_xlabel('Channel [-]')
ax.set_ylabel('Signal [-]')
ax.legend()


# SVD inversion and retrofit
fig_svd = plt.figure()
gs_svd = fig_svd.add_gridspec(1, 3, width_ratios=[1, 0.05, 1], wspace=0.1)
ax1 = fig_svd.add_subplot(gs_svd[0, 2])

ax1.plot(signal, label='Signal')
ax1.plot(retrofit, label='Retrofit')
ax1.set_xlabel('Channel [-]')
ax1.set_ylabel('Signal [-]')
ax1.legend()

ax2 = fig_svd.add_subplot(gs_svd[0, 0])
ax2.set_title('SVD')
im2 = ax2.imshow(tmp.reshape(grid.shape), origin='lower', extent=grid.extent)
ax2.set_xlabel('R [-]')
ax2.set_ylabel('z [-]')


# GEV inversions with different fast regularisation parameter selection methods
vmin = min(out_gev.min(), out_gev_quantile.min(), out_gev_logm.min())
vmax = max(out_gev.max(), out_gev_quantile.max(), out_gev_logm.max())

fig_gev = plt.figure()
fig_gev.suptitle('GEV fast regularisation methods')
gs = fig_gev.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.1)
ax1 = fig_gev.add_subplot(gs[0, 0])
ax1.label_outer()
ax2 = fig_gev.add_subplot(gs[0, 1])
ax2.label_outer()
ax3 = fig_gev.add_subplot(gs[0, 2])
ax3.label_outer()

ax1.set_title('Median')
im1 = ax1.imshow(out_gev.reshape(grid.shape), origin='lower', extent=grid.extent, vmin=vmin, vmax=vmax)
ax1.set_xlabel('R [-]')
ax1.set_ylabel('z [-]')

ax2.set_title('Quantile')
im2 = ax2.imshow(out_gev_quantile.reshape(grid.shape), origin='lower', extent=grid.extent, vmin=vmin, vmax=vmax)
ax2.set_xlabel('R [-]')

ax3.set_title('Log-mean')
im3 = ax3.imshow(out_gev_logm.reshape(grid.shape), origin='lower', extent=grid.extent, vmin=vmin, vmax=vmax)
ax3.set_xlabel('R [-]')

cax = ax3.inset_axes([1.05, 0, 0.05, 1])
fig_gev.colorbar(im1, cax=cax, label='Emissivity [-]')


plt.show()
