# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Template class for general tomographic diagnostic system.
"""
import warnings

import h5py
import numpy as np

from pathlib import Path


class Dsystem(object):
    """
    Contains basic information about diagnostic system

    Attributes
    ----------
    chnl_index : dict
        dict maping channel to detector
    geometry_path : pathlib.Path
        path to location with saved geometrical data
    name : str
        name of diagnostic system
    n_dets : int
        number of cameras with separate calibration files, (to be removed)
    """

    def __init__(self, shot, geometry_path='', **kw):
        super().__init__()
        self.name = self.__class__.__name__
        self.shot = shot
        self.geometry_path = Path(geometry_path)
        self.chnl_index = {}
        self.calb_0 = None
        self.n_dets = None
        self.keywords = kw
        return

    @property
    def data_name(self):
        """
        Template data file name
        """
        name = type(self).__name__
        return '{nm}_{shot}'.format(nm=name, shot=self.shot)

    def download_data(self, **kwargs):
        """
        Downloads data from network/database using `source` method or database.

        Returns
        -------
        dict
        """
        raise NotImplementedError("Download data not defined in subclass")

    def load_data(self, loc):
        """
        Loads hdf file containing data.

        If loc is path to a directory, default data file name from property data_name is assumed.

        Parameters
        ----------
        loc : str or Path
            Path to file with signal data. Can be full path or path to folder if default name is used.

        Returns
        -------
        dict
            keys values, chnl, time
        """
        loc = Path(loc)
        data = dict()
        if loc.is_dir():
            loc = Path(loc) / self.data_name
        with h5py.File(str(loc), 'r') as array:
            for key in array:
                data[key] = array[key][...]
        return data

    # TODO integrate into fetch and remove?
    @staticmethod
    def save_data(loc, data):
        """
        Saves downloaded detector data to data_path

        Parameters
        ----------
        loc : str or pathlib.Path
            saving path with file name
        data : dict
            contains signal values, channel numbers and time stamps
        """
        loc = Path(loc)
        try:
            loc.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(str(loc), 'w') as hdf:
                for key in ['values', 'chnl', 'time']:
                    hdf[key] = data[key]
        except OSError as e:
            warnings.warn('Detector data were not saved: {}'.format(e))
        return

    def fetch_data(self, data_path='', source=None, save=False, force_dl=False):
        """
        Loads data from data storage specified by data_path or downloads data
        from network storage.

        Parameters
        ----------
        data_path : str or Path, optional
            default is working directory
        source : :obj:`string`, optional
            specifies method for data download
        save : bool, optional
            switches saving to local storage
        force_dl : bool, optional
            forces download of data

        Returns
        -------
        dict
        """
        floc = Path(data_path)
        try:
            assert force_dl is False
            data = self.load_data(floc)
        except (OSError, AssertionError):
            data = self.download_data(source=source)
            if save:
                self.save_data(floc / self.data_name, data)
        return data

    def get_calb(self):
        """
        Loads preset calibrations for detectors, if not specified returns array of ones with length n_dets.
        """
        warnings.warn(
            'Method get_calb not specified in Dsystem subclass. ' +
            'Using default get_calb, which returns calb_0 or ones.',
            Warning)
        # TODO add method for loading detector calibrations
        if self.calb_0 is None:
            self.calb_0 = np.ones(self.n_dets)
        return self.calb_0

    def load_los(self, loc=None):
        """
        Loads line of sights geometry from hdf file.

        Parameters
        ----------
        loc : str or Path, optional
            location of los file, by default geometry path is used

        Returns
        -------
        dict
            with coordinates of los start and end points with shape (2, #chords, 3)
        """
        los = {}
        if loc is None:
            los_path = self.geometry_path / self.name
        else:
            los_path = Path(loc)
        with h5py.File(los_path, 'r') as fl:
            for key in fl.keys():
                st = fl[key]['start'][:]
                dir = fl[key]['dirs'][:]
                los[key] = np.stack((st, st+dir), axis=0)
        return los
    
    def get_chord_geometry(self):
        """
        Loads chord geometry from `geometry_path`.

        Expected names of detectors are detector_[num]_[x,y].txt.
        Data for x and y coordinates are in separate files.
        Each row represents one detector.
        Number of rows and columns should be same in all files containing coordinates

        Returns
        -------
        xchords, ychords : numpy.ndarray
            arrays with coordinates of chord points with shape (#chords, #points)
        """
        warnings.warn('Support for LoS saving to txt will be removed. Transfer your coordinates to hdf.', FutureWarning)
        geometry_path = self.geometry_path
        ychords = np.zeros((0, 0))
        xchords = np.zeros((0, 0))
        nl = 0
        i = 0
#        print(geometry_path / 'detector_{}_x.txt'.format(i))
        while (geometry_path / 'detector_{}_x.txt'.format(i)).exists():
            xchord = np.loadtxt(str(geometry_path / 'detector_{}_x.txt'.format(i)))
            if xchord.shape[1] == 3:
                xchord[:, 1] = np.average(xchord[:, :2], axis=1)
                xchord = xchord[:, 1:3]
            try:
                xchords = np.append(xchords, xchord, axis=0)
            except ValueError:
                xchords = xchords.reshape(0, xchord.shape[1])
                xchords = np.append(xchords, xchord, axis=0)
                ychords = ychords.reshape(0, xchord.shape[1])
            ychord = np.loadtxt(str(geometry_path / 'detector_{}_y.txt'.format(i)))
            if ychord.shape[1] == 3:
                ychord[:, 1] = np.average(ychord[:, :2], axis=1)
                ychord = ychord[:, 1:3]
            ychords = np.append(ychords, ychord, axis=0)
            self.chnl_index[i] = np.arange(xchord.shape[0]) + nl
            nl = nl + xchord.shape[0]
            i += 1
        if nl == 0:
            raise IOError('Could not import any geometric chords.' +
                          'Check geometry path and format of geometry data. ' +
                          str(geometry_path / 'detector_{}_x(y).txt'.format(i))
                          )
        return xchords, ychords

    def get_pinholes(self):
        """
        Loads pinhole coordinates for each chord from `pinholes.txt` located
        in geometry folder.

        Returns
        -------
        ph : numpy.ndarray
            matrix containing pinhole x,y coordinates with shape (#chords, 2)
        """
        geometry_path = self.geometry_path
        ph = np.loadtxt(str(geometry_path / 'pinholes.txt'))
        return ph

    def get_chord_widening(self):
        """
        Tries to load widening coefficients from file. Returns None if IOError

        Returns
        -------
        array-like or None
            Contains widening coefficients for each channel

        """
        geometry_path = self.geometry_path
        try:
            widths = np.loadtxt(str(geometry_path / 'widening.txt'))
        except IOError:
            widths = None
        return widths

    def load_boundary_coord(self):
        """
        Load border coordinates from geometry path or from tokamak module.
        Sets boundary_coord attribute to loaded coords.

        Returns
        -------
        numpy.ndarray
            contains (R, z) coordinates of vacuum vessel cross section points, should have shape (#points, 2)
        """
        warnings.warn('load_boundary_coord will be deprecated by load_border', FutureWarning)
        bdr_path = self.geometry_path / 'border.txt'
        boundary_coord = self.load_border(self, bdr_path)
        if np.all(boundary_coord[:, 0] != boundary_coord[:, -1]):
            boundary_coord = np.vstack([boundary_coord, boundary_coord[0, :][np.newaxis, :]])
        return boundary_coord


    def load_border(self, loc=None):
        """
        Load border coordinates from geometry path or from tokamak module.
        Sets boundary_coord attribute to loaded coords.

        Parameters
        ----------
        loc : str or Path

        Returns
        -------
        numpy.ndarray
            contains (R, z) coordinates of vacuum vessel cross section, shape (#points, 2)
        """
        if loc is None:
            bdr_path = self.geometry_path / 'border.txt'
        else:
            bdr_path = Path(loc)
        if not bdr_path.exists():
            print('border coord not found at {}.'.format(bdr_path))
            bdr_path = self.geometry_path.parent / 'border.txt'
            print('Tyring to load from {}'.format(bdr_path))
        bdr = np.loadtxt(str(bdr_path))
        return bdr

    # TODO move to Pixgrid?
    def compute_bd_mat(self, grid, loc=None):
        """
        Simple generation of boundary matrix 1 inside rec area, 0 outside

        Parameters
        ----------
        grid : RegularGrid

        Returns
        -------
        boundary_matrix : numpy.ndarray of bool
            matrix with true inside border, false otherwise, with shape (grid.nz, grid.nr)
        """
        limiter = self.load_border(loc)
        return grid.is_inside(limiter[:, 0], limiter[:, 1])
