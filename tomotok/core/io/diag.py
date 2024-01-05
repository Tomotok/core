# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Template class for general tomographic diagnostic system.
"""
import json
import warnings
from pathlib import Path
from typing import Tuple, Union

import numpy as np


class Dsystem(object):
    """
    Contains basic information about diagnostic system

    Attributes
    ----------
    chnl_index : dict
        dict maping channel to detector
    geometry_path : pathlib.Path
        path to location containing diagnostics system geometry data
    name : str
        name of diagnostic system
    n_dets : int
        number of cameras with separate calibration files, (to be removed)
    """

    def __init__(self, shot, geometry_path: Union[str, Path] = '', **kw):
        """
        Parameters
        ----------
        shot : int
            shot number
        geometry_path : str or pathlib.Path, optional
            path to geometry folder, by default '' (current folder)
        """
        super().__init__()
        self.name = self.__class__.__name__
        self.shot = shot
        self.geometry_path = Path(geometry_path).expanduser()
        self.chnl_index = {}
        self.keywords = kw
        return

    def download_data(self, **kwargs) -> dict:
        """
        Downloads data from network/database using `source` method or database.

        Returns
        -------
        dict
            holds time axis and data
        """
        raise NotImplementedError("Download data should be defined in a subclass.")

    def load_los(self, loc: Union[str, Path] = None, check_data: bool = False) -> dict:
        """
        Loads line of sights geometry from json file.

        Parameters
        ----------
        loc : str or Path, optional
            location of los file, by default geometry path is used

        Returns
        -------
        dict
            a key for each camera that holds dictionary with `startpoints` and `endpoints` keys
            each coordinates list has shape (#chords, 3)
        """
        los = {}
        if loc is None:
            los_path = self.geometry_path / f'{self.name}_los.json'
        else:
            los_path = Path(loc).expanduser()
        with open(los_path, 'r') as fl:
            los = json.load(fl)
        if not isinstance(los, dict):
            raise ValueError('los file must contain dictionary')
        if check_data:
            for key in los:
                if 'startpoints' not in los[key] or 'endpoints' not in los[key]:
                    raise ValueError('Los file must contain startpoints and endpoints for each detector.')
                if len(los[key]['startpoints']) != len(los[key]['endpoints']):
                    raise ValueError('Start and end points lists must have same length and shape.')
                sp_coords_fail = any([len(sp) != 3 for sp in los[key]['startpoints']])
                ep_coords_fail = any([len(ep) != 3 for ep in los[key]['endpoints']])
                if sp_coords_fail or ep_coords_fail:
                    raise ValueError('Start and end points arrays must provide 3D Cartesian coordinates for each point.')
        return los

    def get_chord_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads chord geometry from `geometry_path`.

        Expected names of detectors are detector_[num]_[x,y].txt.
        Data for x and y coordinates are in separate files.
        Each row represents one detector.
        Number of rows and columns should be same in all files containing coordinates

        .. deprecated:: 2.0
            use :meth:`load_los` instead

        Returns
        -------
        xchords, ychords : numpy.ndarray
            arrays with coordinates of chord points with shape (#chords, #points)
        """
        warnings.warn('x/y chord format was deprecated. Use start/end points', DeprecationWarning)
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

        .. deprecated:: 2.0
            no clear universal use case

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

        .. deprecated:: 2.0
            no clear universal use case

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

    def load_border(self, shot: int = None, **kw) -> np.ndarray:
        """
        Load border coordinates from geometry path or from tokamak module.
        Sets boundary_coord attribute to loaded coords.

        Parameters
        ----------
        shot : int, optional
            shot number used to determine configuration, by default self.shot

        Returns
        -------
        numpy.ndarray
            contains (R, z) coordinates of vacuum vessel cross section, shape (#points, 2)
        """
        raise NotImplementedError()

# TODO: remove
    def compute_bd_mat(self, grid, loc=None):
        """
        Simple generation of boundary matrix 1 inside rec area, 0 outside

        .. deprecated:: 1.3.0
            use :obj:`tomotok.core.grid.RegularGrid.is_inside` instead

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
