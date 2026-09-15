# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import RectBivariateSpline

from tomotok.geometry import Grid, RegularGrid


@dataclass
class Signals:
    """
    Container for signals.
    """
    time_axis: np.ndarray
    channel_numbers: np.ndarray
    values: np.ndarray


@dataclass
class Magnetics:
    """
    Container for magnetics data.
    """
    time_axis: np.ndarray
    vertical: np.ndarray
    radial: np.ndarray
    flux: np.ndarray

    def interpolate(
        self,
        grid: RegularGrid,
        tvec: ArrayLike | None = None,
        unique: bool = False,
    ) -> 'Magnetics':
        """
        Interpolates given magnetic field to provided grid using rectangular bivariate spline.

        If `tvec` is provided, only given time slices are interpolated.
        If `unique` is set to True, only unique time slices specified in `tvec` are interpolated.

        Parameters
        ----------
        grid : RegularGrid
        tvec : array-like, optional
        unique : bool, optional
            If False, the same time slice can be interpolated multiple times to provide field for each value in `tvec`.
            If True, only unique time slices specified in `tvec` are interpolated. 
            Default value is False.

        Returns
        -------
        Magnetics
        """
        if tvec is None:
            tidx = np.arange(self.time_axis.size, dtype=int)
        else:
            tidx = np.searchsorted(self.time_axis, tvec)
            if unique:
                tidx = np.unique(tidx)

        nslices = tidx.size
        mfs = np.zeros((nslices, grid.nz, grid.nr))
        for i in range(nslices):
            flux = self.flux[tidx[i]]
            rbs = RectBivariateSpline(self.vertical, self.radial, flux)
            psi_rbs = rbs(grid.z_center, grid.r_center)
            mfs[i] = psi_rbs
        mf_interpolated = Magnetics(
            flux=mfs,
            time_axis=self.time_axis[tidx],
            radial=grid.r_center,
            vertical=grid.z_center
        )
        return mf_interpolated


@dataclass
class Reconstruction:
    """
    Container for reconstruction results.
    """
    time_axis: np.ndarray
    grid: Grid
    values: np.ndarray


@dataclass
class Sightlines:
    """
    Container for sight line coordinates.

    Attributes
    ----------
    start_points : numpy.ndarray
        Array with sight line start points coordinates, with shape (..., 3).
        Last dimension contains Cartesian coordinates.
    end_points : numpy.ndarray
        Array with sight line end points coordinates, with shape (..., 3).
        Last dimension contains Cartesian coordinates.
    """
    # channel_numbers: np.ndarray
    start_points: np.ndarray
    end_points: np.ndarray

    def coords_r(self, number: int = 3) -> np.ndarray:
        """
        Returns given number of radial coordinates along the sight lines.

        Parameters        
        ----------
        number : int, optional
            Number of points to interpolate along the sight line.
            Default is 3 which is minimum not to overlook tangential sight lines.
        """
        x_start = self.start_points[..., 0]
        x_end = self.end_points[..., 0]
        y_start = self.start_points[..., 1]
        y_end = self.end_points[..., 1]

        x = np.linspace(x_start, x_end, num=number)
        y = np.linspace(y_start, y_end, num=number)
        r = np.sqrt(x**2 + y**2)
        return r

    def coords_z(self, number: int = 3) -> np.ndarray:
        """
        Returns given number of vertical coordinates along the sight lines.

        Parameters        
        ----------
        number : int, optional
            Number of points to interpolate along the sight line.
            Default is 3 which is minimum not to overlook tangential sight lines.
        """
        z_start = self.start_points[..., 3]
        z_end = self.end_points[..., 3]
        return np.linspace(z_start, z_end, num=number)

    def cylindrical_coords(self, number: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns given number of cylindrical coordinates along the sight lines.

        Parameters        
        ----------
        number : int, optional
            Number of points to interpolate along the sight line.
            Default is 3 which is minimum not to overlook tangential sight lines.
        """
        return self.coords_r(number), self.coords_z(number)


# @dataclass
# class GeometryMatrix:
#     """
#     Container for geometry matrix and corresponding grid.

#     Attributes
#     ----------
#     gmat : numpy.ndarray or scipy.sparse.sparray
#         geometry matrix with shape (# channels, # nodes)
#     grid : RegularGrid
#         grid corresponding to geometry matrix
#     """
#     gmat: np.ndarray | 'sparray'
#     grid: RegularGrid
