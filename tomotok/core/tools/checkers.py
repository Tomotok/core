# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Contains classes and functions for user checking of some tomography algorithm parts

Examples
--------
Checking anisotropic derivative matrix computed from DataArray with magnetic flux surfaces `magnetic_flux`

>>> from tomotok.core import RegularGrid, Tokamak
>>> from tomotok.core.derivatives import anisotropic_derivative_matrix
>>> 
>>> grid = RegularGrid(50, 100, (0.5, 1), (-0.5, 0.5))
>>> magnetic_flux = Tokamak.download_mag_field(...)
>>> flux = Tokamak.interpolate_mag_field(magnetic_flux, grid, time)
>>> checker = AnisotropicDerivativeChecker(grid)
>>> dmat_par = anisotropic_derivative_matrix(grid, flux, 'parallel')
>>> dmat_per = anisotropic_derivative_matrix(grid, flux, 'perpendicular')
>>> checker(flux, dmat_par, dmat_per)
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy import sparse

from tomotok.core.geometry import RegularGrid


class AnisotropicDerivativeChecker(object):
    """
    Computes anisotropic matrix for a time slice from provided magnetic flux and plots its components.
    Supports interactive selection of matrix element using mouse.

    Attributes
    ----------
    fig : matplotlib.figure
        contains color mesh and countour plot of magnetic surfaces
    fig2 :  matplotlib.figure
        contains two subplots with parallel and perpendicular parts of derivative matrix
    """
    def __init__(
            self, grid: RegularGrid, flux: np.ndarray, dmat1:sparse.spmatrix, dmat2: sparse.spmatrix,
            contour_colors='k', contour_levels=10, dmat_cmap='RdBu',
        ) -> None:
        """
        Parameters
        ----------
        grid : RegularGrid
        """
        self.grid = grid
        self.fig, self.ax = plt.subplots()
        self.fig2, self.ax2 = plt.subplots(figsize=(8, 5), nrows=1, ncols=2, sharey=True)
        self.ax.set_title('Magnetic flux')
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('R [m]')
        self.ax.set_ylabel('z [z]')
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        im_extent = (-1.5* grid.dr, 1.5 * grid.dr, -1.5 * grid.dz, 1.5 * grid.dz)
        self.ax2[0].set_title('Derivative 1')
        self.ax2[0].set_xlabel(r'r - r${}_0$ [m]')
        self.ax2[0].set_xticks([-grid.dr, 0, grid.dr])
        self.ax2[0].set_ylabel(r'z - z${}_0$ [m]')
        self.ax2[0].set_yticks([-grid.dz, 0, grid.dz])
        norm1 = TwoSlopeNorm(0, dmat1.min(), dmat1.max())
        self.im1 = self.ax2[0].imshow(np.ones((3, 3)), cmap=dmat_cmap, norm=norm1, extent=im_extent)

        self.ax2[1].set_title('Derivative 2')
        self.ax2[1].set_xlabel(r'r - r${}_0$ [m]')
        self.ax2[1].set_xticks([-grid.dr, 0, grid.dr])
        norm2 = TwoSlopeNorm(0, dmat2.min(), dmat2.max())
        self.im2 = self.ax2[1].imshow(np.ones((3, 3)), cmap=dmat_cmap, norm=norm2, extent=im_extent)

        self.annotate1_list = {}
        self.annotate2_list = {}
        ra = [-grid.dr, 0, grid.dr]
        za = [grid.dz, 0, -grid.dz]
        for i, j in np.ndindex((3, 3)):
            self.annotate1_list[(i, j)] = self.ax2[0].annotate('', xy=(ra[j], za[i]), ha='center', va='center')
            self.annotate2_list[(i, j)] = self.ax2[1].annotate('', xy=(ra[j], za[i]), ha='center', va='center')

        self.ax.pcolormesh(*self.grid.center_mesh , flux)
        self.ax.contour(*self.grid.center_mesh, flux, levels=contour_levels, colors=contour_colors)

        nr = self.grid.nr

        self.deriv1 = np.zeros((self.grid.size, 3, 3))
        self.deriv1[nr+1:, 2, 0] = dmat1.diagonal(-nr -1)
        self.deriv1[nr:, 2, 1] = dmat1.diagonal(-nr)
        self.deriv1[nr-1:, 2, 2] = dmat1.diagonal(-nr +1)
        self.deriv1[1:, 1, 0] = dmat1.diagonal(-1)
        self.deriv1[:, 1, 1] = dmat1.diagonal(0)
        self.deriv1[:-1, 1, 2] = dmat1.diagonal(1)
        self.deriv1[:-nr+1, 0, 0] = dmat1.diagonal(nr -1)
        self.deriv1[:-nr, 0, 1] = dmat1.diagonal(nr)
        self.deriv1[:-nr-1, 0, 2] = dmat1.diagonal(nr +1)

        self.deriv2 = np.zeros((self.grid.size, 3, 3))
        self.deriv2[nr+1:, 2, 0] = dmat2.diagonal(-nr -1)
        self.deriv2[nr:, 2, 1] = dmat2.diagonal(-nr)
        self.deriv2[nr-1:, 2, 2] = dmat2.diagonal(-nr +1)
        self.deriv2[1:, 1, 0] = dmat2.diagonal(-1)
        self.deriv2[:, 1, 1] = dmat2.diagonal(0)
        self.deriv2[:-1, 1, 2] = dmat2.diagonal(1)
        self.deriv2[:-nr+1, 0, 0] = dmat2.diagonal(nr -1)
        self.deriv2[:-nr, 0, 1] = dmat2.diagonal(nr)
        self.deriv2[:-nr-1, 0, 2] = dmat2.diagonal(nr +1)

        plt.tight_layout()

        self.fig.canvas.draw()
        self.point = self.ax.plot(*grid.centre, 'r+')[0]
        self.update(*grid.centre)
        return

    def update(self, x, y):
        """
        Updates plots for current pixel coordinate
        """
        r_idx = np.abs(self.grid.r_center - x).argmin()
        z_idx = np.abs(self.grid.z_center - y).argmin()
        r = self.grid.r_center[r_idx]
        z = self.grid.z_center[z_idx]
        self.point.set_data(r, z)

        self.ax2[0].set_title(f'dmat1, $r_0$={r:.3f}, $z_0$={z:.3f}')
        self.ax2[1].set_title(f'dmat2, $r_0$={r:.3f}, $z_0$={z:.3f}')

        ind = int(z_idx * self.grid.nr + r_idx)

        sliced = self.deriv1[ind, :, :]
        self.im1.set_data(sliced)
        self.im1.axes.figure.canvas.draw()

        sliced = self.deriv2[ind, :, :]
        self.im2.set_data(sliced)
        self.im2.axes.figure.canvas.draw()

        for i, j in np.ndindex(self.deriv1[ind, :, :].shape):
            self.annotate1_list[(i, j)].set_text(f'{self.deriv1[ind, i, j]:0.2f}')
            self.annotate2_list[(i, j)].set_text(f'{self.deriv2[ind, i, j]:0.2f}')
        self.fig.canvas.draw()
        self.fig2.canvas.draw()

    def onclick(self, event):
        """
        Gets coordinate from the click on the main flux plot
        """
        x = event.xdata
        y = event.ydata
        if x is None or y is None:  # handle clicks outside axes
            pass
        else:
            self.update(x, y)
        return
