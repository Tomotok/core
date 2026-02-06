# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Contains classes and functions for user checks of selected inputs for tomographic inversion

Examples
--------
Checking anisotropic derivative matrix

>>> from tomotok.core.io import RegularGrid, Tokamak
>>> time = 1.2
>>> shot = 19925
>>> coord = RegularGrid(50, 100, (0.5, 1), (-0.5, 0.5))
>>> magf = Tokamak.interpolate_mag_field(magnetic_flux, coord)
>>> checker = DerivMatChecker(coord, magf)
>>> checker(time)
"""
import matplotlib.pyplot as plt
import numpy as np

from tomotok.core.geometry.grids import RegularGrid
from tomotok.core.derivative import prepare_mag_data, generate_anizo_matrix


class DerivMatChecker(object):
    """
    Computes anisotropic matrix for a time slice from provided magnetic flux and plots its components.
    Supports interactive selection of matrix element using mouse.

    Parameters
    ----------
    grid : RegularGrid
    magfield : dict

    Attributes
    ----------
    fig : matplotlib.figure
        contains color mesh and countour plot of magnetic surfaces
    fig2 :  matplotlib.figure
        contains two subplots with parallel and perpendicular parts of derivative matrix
    """
    def __init__(self, grid: RegularGrid, magfield):
        self.nx, self.ny = grid.nr, grid.nz
        self.coord = grid
        self.fluxes = magfield
        self.fig, self.ax = plt.subplots()
        self.fig2, self.ax2 = plt.subplots(figsize=(8, 5), nrows=1, ncols=2)

        self.ax.set_title('Magnetic flux')
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.ax2[0].set_title('Check par_derivative')
        self.ax2[0].set_xlabel('x [px]')
        self.ax2[0].set_ylabel('y [px]')

        self.ax2[1].set_title('Check per_derivative')
        self.ax2[1].set_xlabel('x [px]')
        self.ax2[1].set_ylabel('y [px]')
        self.ax2[1].yaxis.tick_right()
        return

    def __call__(self, time, der_type=4):
        idx = np.searchsorted(self.fluxes['tvec'], time)
        flux = self.fluxes['values'][idx]
        self.t = time
        self.ax.images.clear()
        self.ax.collections.clear()
        self.ax.imshow(flux, origin='bottom',
                       # extent=self.coord.extent
                       )
        self.ax.contour(flux, np.sqrt(np.linspace(0, 1.0, 20)),
                        # extent=self.coord.extent
                        )

        atan2 = prepare_mag_data(flux[:, :])
        bper, bpar, bpar_dense, bper_dense = generate_anizo_matrix(self.coord, atan2, der_type)

        self.deriv1 = bper_dense.reshape(self.nx * self.ny, 3, 3)[:, :, :]
        self.deriv2 = bpar_dense.reshape(self.nx * self.ny, 3, 3)[:, :, :]
        # Central pixel index
        ind = int(self.coord.nz / 2 * self.coord.nr + self.coord.nr / 2)

        self.slices, rows, cols, = self.deriv1.shape
        sliced = self.deriv1[ind, :, :]
        self.im1 = self.ax2[0].imshow(sliced, cmap='Greys')

        self.slices, rows, cols, = self.deriv2.shape
        sliced = self.deriv2[ind, :, :]
        self.im2 = self.ax2[1].imshow(sliced, cmap='Greys')

        self.ax2[0].texts.clear()
        self.ax2[1].texts.clear()
        self.annotate1_list = {}
        self.annotate2_list = {}

        for i, j in np.ndindex(self.deriv1[ind, :, :].shape):
            # print(ind, i, j)
            annotate1_tmp = self.ax2[0].annotate('{:0.2}'.format(self.deriv1[ind, i, j]), (j, i), color='r')
            annotate2_tmp = self.ax2[1].annotate('{:0.2}'.format(self.deriv2[ind, i, j]), (j, i), color='r')
            self.annotate1_list[(i, j)] = annotate1_tmp
            self.annotate2_list[(i, j)] = annotate2_tmp

        plt.tight_layout()

        self.fig.canvas.draw()
        self.fig2.canvas.draw()
        return

    def update(self, x, y):
        """
        Updates plots for current pixel coordinate
        """
        self.ax2[0].set_title('Bpar, x={},y={}'.format(x, y))
        self.ax2[1].set_title('Bper, x={},y={}'.format(x, y))

        ind = int(y * self.coord.nx + x)

        sliced = self.deriv1[ind, :, :]
        self.im1.set_data(sliced)
        # self.im1.axes.figure.canvas.draw()

        sliced = self.deriv2[ind, :, :]
        self.im2.set_data(sliced)

        for i, j in np.ndindex(self.deriv1[ind, :, :].shape):
            self.annotate1_list[(i, j)].set_text('{:0.2}'.format(self.deriv1[ind, i, j]))
            self.annotate2_list[(i, j)].set_text('{:0.2}'.format(self.deriv2[ind, i, j]))
        self.fig2.canvas.draw()

    def onclick(self, event):
        """
        Gets coordinate from the click on the main flux plot
        """
        # print('Returned indices')
        # print(event.xdata, event.ydata)
        # print('mapping back:')
        x = int(np.round(event.xdata))
        y = int(np.round(event.ydata))
        # tx = "Y: {}, X: {}".format(y, x)
        # print(tx)
        self.update(x, y)
