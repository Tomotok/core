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
from matplotlib.image import imread
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm

from tomotok.core.phantoms import gauss_iso
from tomotok.core.geometry import RegularGrid, sparse_line_3d, generate_los
from tomotok.core.derivative import derivative_matrix
from tomotok.core.inversions import GevFastAlgebraic, Mfr, SimpleBob, SvdFastAlgebraic


rcParams['text.usetex'] = True
rcParams['font.size'] = 6
rcParams['figure.dpi'] = 200
rcParams['image.origin'] = 'lower'
rcParams['image.cmap'] = 'RdBu'
rcParams['lines.linewidth'] = 1

svd = SvdFastAlgebraic()
gev = GevFastAlgebraic()
mfr = Mfr()

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
# grid = RegularGrid(20, 40, rlim, zlim)  # 2x2cm, used for paper figures
grid = RegularGrid(40, 80, rlim, zlim)  # 1x1cm
# grid = RegularGrid(80, 160, rlim, zlim)  # .5x.5cm

# create line of sights for the linear arrays
num = 20  # detectors per array
# horizontal
s1, e1 = generate_los(num=(num, 1), fov=(70, 0), pinhole=(1, 0, 0), axis=(-1, 0, 0))  
# top
s2, e2 = generate_los(num=(num, 1), fov=(50, 0), pinhole=(0.5, 0, 0.7), axis=(0.01, 0, -1), elong=1.5)
# angled bottom
s3, e3 = generate_los(num=(num, 1), fov=(50, 0), pinhole=(0.9, 0, -0.5), axis=(-1, 0, 1), elong=1.5)  
# angled top
s4, e4 = generate_los(num=(num, 1), fov=(50, 0), pinhole=(0.9, 0, 0.5), axis=(-1, 0, -1), elong=1.5)  

# combine line of sights coordinates of arrays into one variable
ss = []
es = []
ss.append(s1)
es.append(e1)
ss.append(s2)
es.append(e2)
ss.append(s3)
es.append(e3)
es.append(e4)
ss.append(s4)
s = np.concatenate(ss, 0)
e = np.concatenate(es, 0)

# reorganise for the geometry matrix computation
chnls = np.arange(s.shape[0])
rch = np.zeros((s.shape[0], 2))
zch = np.zeros((s.shape[0], 2))
rch[:, 0] = s[:, 0]
rch[:, 1] = e[:, 0]
zch[:, 0] = s[:, 2]
zch[:, 1] = e[:, 2]

gmat = sparse_line_3d(rch, zch, grid, rmin=.2)
dgmat = gmat.toarray()

phantom = gauss_iso(grid.nr, grid.nz, w=0.2) * 100

# create synthetic signal
sig = gmat.dot(phantom.flatten())
# add noise
# ampl_noise = sig.max() * 0.02
# sig += np.random.normal(0, ampl_noise, sig.size)

data = sig.reshape(1, -1)  # data should have shape (#time slices, #channels/pixels)

# expected error in data
errors = 0.001
# derivative matrices for the inversion
derivs = [
    [derivative_matrix(grid, 'right'), derivative_matrix(grid, 'top')],
    [derivative_matrix(grid, 'top'), derivative_matrix(grid, 'left')],
]

ela = time.time()
sout = svd(data, dgmat, derivatives=derivs, errors=errors, method='logmean')
ela = time.time() - ela
print('svd', ela, 's')

ela = time.time()
gout = gev(data, dgmat, derivs, errors, method='logmean')
ela = time.time() - ela
print('gev', ela, 's')

mout = mfr(data, gmat, derivs, errors)


# # Matrix Camera
# - sets up a system based on tangentially viewing matrix camera
# - computes geometry matrix
# - creates phantom emissivity 
# - computes synthetic image
# - performs inversions using MFR and BOB

resolution = (80, 80)  # new camera resolution
fov = (60, 60)
pinhole_position = (0.8, 0.1, 0.2)
camera_axis = (-1, -0.2, 0.25,)

start, end = generate_los(
    pinhole=pinhole_position,
    axis=camera_axis,
    num=resolution, 
    fov=fov, 
    elong=3,
)

nch = start.shape[0]
xc = np.zeros((nch, 2))
yc = np.zeros((nch, 2))
zc = np.zeros((nch, 2))
xc[:, 0] = start[:, 0]
xc[:, 1] = end[:, 0]
yc[:, 0] = start[:, 1]
yc[:, 1] = end[:, 1]
zc[:, 0] = start[:, 2]
zc[:, 1] = end[:, 2]

grid2 = RegularGrid(30, 40, (.2, .8), (-.4, .4))
gmat2 = sparse_line_3d(xc, yc, grid2, zc, rmin=.2)

grid_column = RegularGrid(1, 1, (0, 0.2, ), (-.4, .4))
gmat_column = sparse_line_3d(xc, yc, grid_column, zc)
image_column = gmat_column.dot(np.array([1]))

phantom2 = gauss_iso(grid2.nr, grid2.nz, cen=.5) * 100
image = gmat2.dot(phantom2.reshape(-1, 1))
derivs2 = [
    [derivative_matrix(grid2, 'right'), derivative_matrix(grid2, 'top')],
    [derivative_matrix(grid2, 'left'), derivative_matrix(grid2, 'bottom')],
]

bob = SimpleBob()
ela2 = time.time()
bob.decompose(gmat2)
ela2 = time.time() - ela2

bout = bob(image, gmat2)
mout2 = mfr(
    data=image.reshape(1, -1), 
    gmat=gmat2, 
    errors=np.ones((1, image.size))*1e-5,
    derivatives=derivs2,
    bounds=(-15,0),
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

lnaax[2].set_axis_off()
lnaax[2].set_title('Artificial Image')

lnaax[0].plot(rch.T, zch.T, 'k', lw=0.5, alpha=0.5)
rct = Rectangle((grid.rmin, grid.zmin), grid.rmax - grid.rmin, grid.zmax - grid.zmin)
lnaax[0].add_patch(rct, )

# The node projections with a wireframe of COMPASS vessel was calculated separately in CALCAM
# It can not be easily reproduced here, so the image is loaded if available
# Otherwise a simple node projection is drawn
try:
    lnaax[1].set_title('Node Projection')
    node = imread('node_image_plane.png')
    lnaax[1].imshow(node, origin='upper')
    lnaax[1].set_axis_off()
except FileNotFoundError:
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('foo', [(0, 'white'), (1, 'C1')])
    tmp = np.zeros_like(phantom2)
    tmp[18, 17] = 100
    replacement = gmat2 @ tmp.flatten()
    replacement = replacement.reshape(resolution).T > 0
    lnaax[1].imshow(replacement, cmap=cmap, origin='lower')
    lnaax[1].set_axis_off()

ncimg = lnaax[2].imshow(image.reshape(resolution).T, cmap='Blues', origin='lower')
nccax = lnaax[2].inset_axes(bounds=[1.1, 0, 0.05, 1])
lna.colorbar(ncimg, cax=nccax, label='Signal [-]')

lnaax[2].text(3.2/4*resolution[0], 1/4*resolution[1], 'Central column', rotation='vertical')
lnaax[2].contour(image_column.reshape(resolution).T, levels=[0], colors='k', linewidths=0.5)


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
