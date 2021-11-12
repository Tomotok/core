# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Universal tokamak template class for creating machine specific subclasses
"""
from pathlib import Path
from warnings import warn

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

from .diag import Dsystem
from ..geometry import RegularGrid


class Tokamak(Dsystem):
    """
    Handles magnetic field and divertor

    Attributes
    ----------
    name : str
        Name of tokamak
    params : dict
        Keyword arguments provided to initialization
    divertor_coord : tuple of float
        Contains bounds for divertor area. (x1, x2, y1, y2)

    """
    def __init__(self, shot, **kwargs):
        super().__init__(shot, **kwargs)
        # self.name = self.__class__.__name__
        self.divertor_coord = (0, 0, 0, 0)
        return

    @property
    def mag_name(self):
        name = type(self).__name__
        return '{nm}_Magnetics_{shot}'.format(nm=name, shot=self.shot)

    def download_mag_field(self, shot, tvec=None, source=None):
        """
        Dowloads magnetic field data from network.

        Parameters
        ----------
        shot : int
        tvec : list-like, optional
            specifies timeslices to be loaded
        source : string, optional
            specifies source for loading magnetics data

        Returns
        -------
        dict
            Psi normalized stored in dict with keys (values, time, z, r)
        """
        raise NotImplementedError('Method download_mag_field should be defined in subclass.')

    def load_mag_field(self, loc):
        """
        Loads magnetic field from hdf file saved at loc

        If loc is path to a directory, default data file name from property mag_name is assumed.

        Parameters
        ----------
        loc : str or Path
            location of hdf file containing magnetic field

        Returns
        -------
        dict
            Psi normalized stored in dict with keys (values, time, z, R)
        """
        loc = Path(loc)
        if loc.is_dir():
            loc = Path(loc) / self.mag_name
        mag_field = np.load(str(loc))
        return mag_field

    @staticmethod
    def save_mag_field(loc, mag_field):
        """
        Saves magnetic field into hdf file.

        Expects to have name included to mag_loc if magname is omitted.
        
        Parameters
        ----------
        loc : str or pathlib.Path
            Path to file
        mag_field : dict, optional
           DataArray with magnetic field
        """
        loc = Path(loc)
        try:
            loc.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(loc), **mag_field)
        except PermissionError as e:
            warn('Magnetic field was not saved: {}'.format(e))
        return

    def fetch_mag_field(self, shot, mag_path, grid=None, tvec=None, save=False, force_dl=False):
        """
        Loads magnetic field from local storage or database and interpolates it if grid is provided.

        Parameters
        ----------
        shot : int
        mag_path : str or Path
        grid : RegularGrid, optional
            Returned values are interpolated to this grid if provided.
        tvec : numpy.ndarray, optional
            Time vector  of requested  magnetic field for download
        save : bool, optional
        force_dl : bool, optional
            forces download from remote database
        """
        mag_path = Path(mag_path)
        try:
            assert force_dl is False
            mag_loc = mag_path / self.mag_name
            magfield = self.load_mag_field(mag_loc)
        except (TypeError, FileNotFoundError, OSError, AssertionError):
            magfield = self.download_mag_field(shot, tvec)
            if save:
                try:
                    mag_path.mkdir(exist_ok=True)
                    self.save_mag_field(mag_path / self.mag_name, magfield)
                except OSError as e:
                    warn('Magnetics data were not saved: {}'.format(e))
        print('Magnetic field loaded.')
        if grid is not None:
            magfield = self.interpolate_mag_field(magfield, grid, tvec)
            print('Magnetic field interpolated.')
        return magfield

    @staticmethod
    def interpolate_mag_field(magfield, grid, tvec=None):
        """
        Interpolates given magnetic field to provided grid using rectangular bivariate spline.

        If tvec is provided, only one nearest time slice of magnetic field is interpolated for each value in tvec.

        Parameters
        ----------
        magfield : dict
        grid : RegularGrid
        tvec : int, float, list, tuple, np.ndarray, optional

        Returns
        -------
        dict
        """
        if tvec is None:
            tidx = np.arange(magfield['time'].size, dtype=np.int)
        else:
            tidx = np.searchsorted(magfield['time'], tvec)
            tidx = np.unique(tidx)
            
        nslices = tidx.size
        mfs = np.zeros((nslices, grid.nz, grid.nr))
        for i in range(nslices):
            flux = magfield['values'][tidx[i]]
            rbs = RectBivariateSpline(magfield['z'], magfield['r'], flux)
            psi_rbs = rbs(grid.z_center, grid.r_center)
            mfs[i] = psi_rbs
        mf_interpolated = dict()
        mf_interpolated['values'] = mfs
        mf_interpolated['time'] = magfield['time'][tidx]
        mf_interpolated['r'] = grid.r_center
        mf_interpolated['z'] = grid.z_center
        return mf_interpolated

    # TODO move to RegularGrid?
    def get_divertor_area(self, grid, boundary=None):
        """
        Returns mask matrix with True for indexes representing pixels inside 
        predefined divertor area. Size and shape is obtained from :attr:`grid`.

        Parameters
        ----------
        grid : RegularGrid
        boundary : numpy.ndarray, optional
            Boundary matrix with same shape as grid

        Returns
        -------
        numpy.ndarray
            Bool mask matrix for determining divertor pixels with shape ny, nx
        """
        xgrid, ygrid = np.meshgrid(grid.r_center, grid.z_center)
        dcoord = self.divertor_coord
        divertor_area = np.logical_and(xgrid > dcoord[0], xgrid < dcoord[1])
        divertor_area *= np.logical_and(ygrid > dcoord[2], ygrid < dcoord[3])
        if boundary is not None:
            divertor_area = divertor_area * boundary
        return divertor_area

    # TODO move to solver or RegularGrid?
    def get_divertor_chnls(self, gmat, grid, boundary=None):
        """
        Finds channels of geometry matrix that have contributions in divertor area.
        Uses dot product of geometry and divertor area matrices.

        Parameters
        ----------
        gmat : csr_matrix
        grid : RegularGrid
        boundary : numpy.ndarray, optional
            Boundary matrix with same shape as grid

        Returns
        -------
        mask : numpy.ndarray of bool
            mask array with True on channels interfering with predefined divertor area
        """
        div_vec = self.get_divertor_area(grid, boundary).flatten()
        gval = gmat.values.reshape(gmat.chnl.size, -1)
        dotp = gval.dot(div_vec)
        mask = dotp.sum(axis=1) > 0
        return mask

    # TODO relocate to algorithms?
    def compute_weight_matrix(self, grid, diw=0.4, smooth_boundary=None, bdr=None):
        """
        Computes weight matrix. Supports different weight for pixels located in divertor area or those that are
        outside vacuum vessel.

        Parameters
        ----------
        grid : RegularGrid
        diw : float, optional
            Specifies divertor pixels' weight. Default value for divertor is 0.4 and standard weight of pixel is 1.
            If standard weight is used, the divertor area will not have any preference.
        smooth_boundary : float, optional
            Specifies weight for pixels outside of vacuum vessel, that is used for smoothing the boundary.
            Vacuum vessel coordinates are defined by bdr parameter. Recommended value is 0.1
        bdr : tuple of numpy.ndarray
            Contains coordinates of vacuum vessel border. Must be provided if smooth_boundary is requested.

        Returns
        -------
        numpy.ndarray dtype float64
            matrix with pixel weights with same shape as grid (#y pixels, #x pixels)
        """
        nx = grid.nr
        ny = grid.nz
        wm = np.ones((ny, nx))
        if diw != 1:
            divertor_area = self.get_divertor_area(grid)
            wm[divertor_area] = diw
        if smooth_boundary:
            bdm = grid.is_inside(*bdr)
            wm = gaussian_filter(wm, sigma=3, mode='nearest')
            wm = np.where(bdm, wm, smooth_boundary)
        return wm
