"""
Performs benchmarking of the inversion algorithms on the synthetic data
as shown in the ECPD 2021 paper

J. Svoboda et al., "Tomotok: python package for tomography of tokamak plasma radiation",
Journal of Instrumentation 16.12 (2021): C12015.

All four figures showing results in the paper are generated and can be saved as images.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy import sparse

from tomotok.core.geometry import RegularGrid, sparse_line
from tomotok.core.regularisations.derivatives import derivative_matrix
from tomotok.core.inversions import Bob, Tikhonov, PearsonSelector
from tomotok.core.inversions.lame import GevAlgebraic, SvdAlgebraic, FastSelector
from tomotok.core.inversions.mfr import MinimumFisherRegularisation
from tomotok.core.regularisations.matrices import weighted_squares
from tomotok.tools.phantoms import regular_elliptical_flux, gaussian_on_flux
from tomotok.tools.sightlines import generate_sightlines


rcParams['text.usetex'] = True
rcParams['font.size'] = 6
rcParams['figure.dpi'] = 200
rcParams['image.origin'] = 'lower'
rcParams['image.cmap'] = 'RdBu'
rcParams['lines.linewidth'] = 1

save = False

# # Linear Detectors
#  - sets up a system of linear array detectors
#  - computes geometry matrix
#  - creates phantom emissivity 
#  - computes synthetic signals 
#  - performs inversions using SVD and GEV variants of Fast Algebraic version of LAME algorithm
#  - performs inversion using MFR 

# grid limits for linear system
rlim = (0.3, 0.7)
zlim = (-0.4, 0.4)

# create inversion grid with desired node size
# grid = RegularGrid(10, 20, rlim, zlim)  # 4x4cm
grid = RegularGrid(20, 40, rlim, zlim)  # 2x2cm, used for paper figures
# grid = RegularGrid(40, 80, rlim, zlim)  # 1x1cm
# grid = RegularGrid(80, 160, rlim, zlim)  # .5x.5cm

# create line of sights for the linear arrays
num = 20  # detectors per array
# horizontal
s1, e1 = generate_sightlines(num=(num, 1), fov=(70, 0), pinhole=(1, 0, 0), axis=(-1, 0, 0))  
# top
s2, e2 = generate_sightlines(num=(num, 1), fov=(50, 0), pinhole=(0.5, 0, 0.7), axis=(0.01, 0, -1), length=1.5)
# angled bottom
s3, e3 = generate_sightlines(num=(num, 1), fov=(50, 0), pinhole=(0.9, 0, -0.5), axis=(-1, 0, 1), length=1.5)  
# angled top
s4, e4 = generate_sightlines(num=(num, 1), fov=(50, 0), pinhole=(0.9, 0, 0.5), axis=(-1, 0, -1), length=1.5)  

# combine line of sights coordinates of arrays into one variable
startpnts = np.concatenate([s1, s2, s3, s4], 0)
endpnts = np.concatenate([e1, e2, e3, e4], 0)

# reorganise for the geometry matrix computation

gmat = sparse_line(startpnts, endpnts, grid, rmin=.2)
dgmat = gmat.toarray()

flux = regular_elliptical_flux(grid, span=1.2)
phantom = gaussian_on_flux(flux, width=0.2, amplitude=100)

# create synthetic signal
sig = gmat.dot(phantom.flatten())
# add noise
# ampl_noise = sig.max() * 0.02
# sig += np.random.normal(0, ampl_noise, sig.size)

# expected error in data
errors = 0.001
# derivative matrices for the inversion
derivs = [
    derivative_matrix(grid, 'right', compensate_edges=False),
    derivative_matrix(grid, 'top', compensate_edges=False),
    derivative_matrix(grid, 'top', compensate_edges=False),
    derivative_matrix(grid, 'left', compensate_edges=False),
]
regularisation = weighted_squares(derivs)

svd = SvdAlgebraic(num=None, regularisation_selector=FastSelector(method='logmean'))
ela = time.time()
sout = svd(sig, dgmat, regularisation, errors=errors)
ela = time.time() - ela
print('svd', ela, 's')

gev = GevAlgebraic(num=None, regularisation_selector=FastSelector(method='logmean'))
ela = time.time()
gout = gev(sig, gmat, regularisation, errors=errors)
ela = time.time() - ela
print('gev', ela, 's')

mfr = MinimumFisherRegularisation(Tikhonov(regularisation_selector=PearsonSelector(bounds=(-15, 0), iter_max=20)))
mout = mfr(sig, gmat, derivs, errors)


# # Matrix Camera
# - sets up a system based on tangentially viewing matrix camera
# - computes geometry matrix
# - creates phantom emissivity 
# - computes synthetic image
# - performs inversions using MFR and BOB

resolution = (80, 80)  # new camera resolution
fov = (60, 60)
pinhole_position = (0.8, 0.2, 0.1)
camera_axis = (-1, 0.25, -0.2,)

start, end = generate_sightlines(
    pinhole=pinhole_position,
    num=resolution, 
    fov=fov, 
    axis=camera_axis,
    length=3,
)

grid2 = RegularGrid(30, 40, (.2, .8), (-.4, .4))
gmat2 = sparse_line(start, end, grid2, rmin=.2)

grid_column = RegularGrid(1, 1, (0, 0.2, ), (-.4, .4))  # single node grid for the column projection
gmat_column = sparse_line(start, end, grid_column)
image_column = gmat_column @ np.array([1])

flux2 = regular_elliptical_flux(grid2, span=1.2)
phantom2 = gaussian_on_flux(flux2, center=0.5, amplitude=100)
image = gmat2.dot(phantom2.reshape(-1, 1))
derivs2 = [
    derivative_matrix(grid2, 'right'),
    derivative_matrix(grid2, 'top'),
    derivative_matrix(grid2, 'left'),
    derivative_matrix(grid2, 'bottom'),
]

basis = sparse.diags_array([1], shape=(grid2.size, grid2.size))
bob = Bob()
ela2 = time.time()
bob.decompose(gmat2.tocsc(), basis)
ela2 = time.time() - ela2

bout = bob(image, gmat2)

mfr2 = MinimumFisherRegularisation(Tikhonov(regularisation_selector=PearsonSelector(bounds=(-10, 1), iter_max=20)))
mout2 = mfr2(
    data=image.flatten(), 
    gmat=gmat2, 
    errors=np.ones((image.size))*1e-5,
    derivatives=derivs2,
)


# # Paper figures


# ## Figure 2
# layout of linear array detectors + a node projection in compass tokamak from CALCAM + artificial image obtained from the matrix camera

lna = plt.figure(figsize=(6, 1.5))
lnaax = lna.subplots(1, 3)


lnaax[0].set_aspect(1)
lnaax[0].set_xlim(0, 1)
lnaax[0].set_xticks((0, grid.rmin, grid.rmax, 1))
lnaax[0].set_ylim(-0.5, 0.7)
lnaax[0].set_title('Linear Layout')
lnaax[0].set_xlabel('R [-]')
lnaax[0].set_ylabel('z [-]')

# lnaax[2].set_axis_off()
lnaax[2].set_xticks([])
lnaax[2].set_yticks([])
lnaax[2].set_title('Artificial Image')

lnaax[0].plot(np.vstack((startpnts[:, 0], endpnts[:, 0])), np.vstack((startpnts[:, 2], endpnts[:, 2])), 'k', lw=0.5, alpha=0.5)
rct = Rectangle((grid.rmin, grid.zmin), grid.rmax - grid.rmin, grid.zmax - grid.zmin)
lnaax[0].add_patch(rct, )

# The node projections with a wireframe of COMPASS vessel was calculated separately in CALCAM
# It can not be made public, therefore, a replacement with approximate central column projection is shown
cmap = LinearSegmentedColormap.from_list('node', [(0, 'white'), (1, 'C1')])
tmp = np.zeros_like(phantom2)
tmp[18, 17] = 1  # selected node with unit emissivity
replacement = gmat2 @ tmp.flatten()  # image of the node
replacement = replacement.reshape(resolution) > 0  # 1 for pixels that observe the node
lnaax[1].imshow(replacement, cmap=cmap, origin='lower')
lnaax[1].text(3/40*resolution[0], 10/40*resolution[1], 'Central column', rotation='vertical')
lnaax[1].contour(image_column.reshape(resolution), levels=[0.01], colors='k', linewidths=0.5)

lnaax[1].set_xticks([])
lnaax[1].set_yticks([])

ncimg = lnaax[2].imshow(image.reshape(resolution), cmap='Blues', origin='lower')
nccax = lnaax[2].inset_axes(bounds=[1.1, 0, 0.05, 1])
lna.colorbar(ncimg, cax=nccax, label='Signal [-]')

lnaax[2].text(3/40*resolution[0], 10/40*resolution[1], 'Central column', rotation='vertical')
lnaax[2].contour(image_column.reshape(resolution), levels=[0.01], colors='k', linewidths=0.5)


# ## Figure 3
# Results of MFR + LAME compared with phantom

resf1 = plt.figure(figsize=(4, 1.5), dpi=200)
resax = resf1.subplots(1, 4, sharey=True, gridspec_kw={'wspace': .05})

datas = [phantom, mout[0], sout[0], gout[0]]
titles = ['Phantom', 'MFR', 'SVD', 'GEV']

vmin = min([d.min() for d in datas])
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=100)

resax[0].set_ylabel('z [-]')
for i in range(4):
    resax[i].set_title(titles[i])
    resax[i].set_xlabel('R [-]')
    # resax[i].set_ylabel('z [-]')
    resax[i].set_xticks((grid.rmin, grid.r_center.mean(), grid.rmax))

    resimg = resax[i].imshow(
        datas[i].reshape(grid.shape), 
        extent=grid.extent, 
        norm=norm,
        # cmap='cividis',
    )

rescax = resax[-1].inset_axes(bounds=[1.3, 0, 0.05, 1])

resfcbar = resf1.colorbar(
    resimg, cax=rescax, label='Emissivity [-]', 
    ticks=[-10, -5, 0, 50, 100],
    spacing='proportional',
    )


# ## Figure 4
# Horizontal and vertical cuts through the phantom and the reconstructions using linear system

col = grid.nr // 2 - 1
row = grid.nz // 2 

cutf = plt.figure(figsize=(6,1.5))
cutax = cutf.subplots(1, 3)

cutax[0].set_title('Phantom')
cutax[0].set_xlabel('R [-]')
cutax[0].set_ylabel('z [-]')
cutax[0].set_xticks((grid.rmin, grid.r_center.mean(), grid.rmax))
cutax[0].imshow(phantom, cmap='Blues', extent=grid.extent)
cutax[0].axhline(grid.z_border[row], lw=1, color='k', ls='--')
cutax[0].axvline(grid.r_border[col+1], lw=1, color='k', ls='--')

cutax[1].set_title('Horizontal')
cutax[1].set_xlabel('R [-]')
cutax[1].set_ylabel('Emissivity [-]')
lnp, = cutax[1].plot(grid.r_center, phantom[row], label='Phantom', lw=1)
lnm, = cutax[1].plot(grid.r_center, mout[0].reshape(grid.shape)[row], label='MFR', lw=1)
lng, = cutax[1].plot(grid.r_center, gout[0].reshape(grid.shape)[row], label='GEV', lw=1)
lns, = cutax[1].plot(grid.r_center, sout[0].reshape(grid.shape)[row], label='SVD', lw=1)

cutax[2].set_title('Vertical')
cutax[2].yaxis.tick_right()
cutax[2].set_ylabel('z [-]')
cutax[2].set_xlabel('Emissivity [-]')
cutax[2].yaxis.set_label_position('right')
cutax[2].plot(phantom[:, col], grid.z_center, label='Phantom', lw=1)
cutax[2].plot(mout[0].reshape(grid.shape)[:, col], grid.z_center, label='MFR', lw=1)
cutax[2].plot(gout[0].reshape(grid.shape)[:, col], grid.z_center, label='GEV', lw=1)
cutax[2].plot(sout[0].reshape(grid.shape)[:, col], grid.z_center, label='SVD', lw=1)

cutf.legend(handles=[lnp, lnm, lng, lns], bbox_to_anchor=[0.58, .5], loc='center left', ncol=1)


# ## Figure 5
# Results of matrix camera setup of MFR and BOB

resf2 = plt.figure(figsize=(5, 1.5), dpi=200)
resax2 = resf2.subplots(1, 3)

r2titles = ['Phantom', 'MFR', 'BOB']

norm2 = TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=100)
kw2 = dict(extent=grid2.extent, norm=norm2)
kwcbar2 = dict(label='Emissivity [-]', ticks=[-0.4, -0.2, 0, 50, 100])

for i, ax in enumerate(resax2):
    ax.set_xlabel('R [-]')
    ax.set_ylabel('z [-]')
    ax.set_title(r2titles[i])
    ax.set_xticks((grid2.rmin, grid2.r_center.mean(), grid2.rmax))

resax2[0].imshow(phantom2, **kw2)
resax2[1].imshow(mout2[0].reshape(grid2.shape), **kw2)
res2img = resax2[2].imshow(bout.reshape(grid2.shape), **kw2)

res2cax = resax2[2].inset_axes(bounds=[1.3, 0, 0.05, 1])
resf2cbar = resf2.colorbar(res2img, cax=res2cax, **kwcbar2)


if save:
    lna.savefig('fig2-setups.png', bbox_inches='tight', pad_inches=0)  # figure 2 layout, projection, image
    resf1.savefig('fig3-res1.png', bbox_inches='tight', pad_inches=0)  # figure 3 linear results
    cutf.savefig('fig4-cuts.png', bbox_inches='tight', pad_inches=0)  # figure 4 cuts through linear results
    resf2.savefig('fig5-res2.png', bbox_inches='tight', pad_inches=0)  # figure 5 matrix camera results


plt.show()
