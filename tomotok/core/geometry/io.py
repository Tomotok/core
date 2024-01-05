# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Saving and loading functions for geometry matrices in sparse format using h5py module.
"""
from typing import Tuple, Union
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sparse

from .grids import RegularGrid


def save_sparse_gmat(floc: Union[str, Path], gmat: sparse.csr_matrix, grid: RegularGrid, attrs_dct: dict = None) -> None:
    """
    Saves geometry matrix together with description of the grid.

    Currently suports only sparse CSR matrices and regular rectangular grids.

    Parameters
    ----------
    floc : str
        file location
    gmat : csr_matrix
        with shape (#chanels, #nodes)
    grid : RegularGrid
        class describing regular grid the gmat was computed for
    attrs_dict : dict, optional
        additional attributes to be saved in hdf file
    """
    floc = Path(floc)
    floc = floc.expanduser()
    if not isinstance(gmat, sparse.csr_matrix):
        msg = 'This function can only save sparse gmat. ' + \
            'For saving matrix in dense format use save_dense_gmat function'
        raise ValueError(msg)
    with h5py.File(floc, 'w') as f:
        f.attrs['version'] = '0.1'
        f.create_dataset('indices', data=gmat.indices)
        f.create_dataset('indptr', data=gmat.indptr)
        f.create_dataset('data', data=gmat.data)
        f.attrs['format'] = gmat.format
        f.attrs['shape'] = gmat.shape
        grp = f.create_group('grid')
        grp.create_dataset('nr', data=grid.nr)
        grp.create_dataset('nz', data=grid.nz)
        grp.create_dataset('rlims', data=grid.rlims)
        grp.create_dataset('zlims', data=grid.zlims)
        grp.attrs['type'] = 'regular_rectangles'
        if attrs_dct is not None:
            for key in attrs_dct:
                f.attrs[key] = attrs_dct[key]


def load_sparse_gmat(floc: Union[str, Path]) -> Tuple[sparse.csr_matrix, RegularGrid]:
    """
    Loads hdf file and creates geometry matrix in sparse csr_matrix format and RegularGrid class

    Parameters
    ----------
    floc : str
        hdf file location
    
    Returns
    -------
    gmat : sparse.csr_matrix
    grid : geometry.RegularGrid
    """
    floc = Path(floc)
    floc = floc.expanduser()
    with h5py.File(floc, 'r') as fl:
        if fl.attrs['format'] != 'csr':
            msg = 'This function can only load sparse gmat. ' + \
                'For loading matrix in dense format use load_dense_gmat function'
            raise ValueError(msg)
        data = fl['data'][:]
        indptr = fl['indptr'][:]
        indices = fl['indices'][:]
        shp = fl.attrs['shape']
        nx = fl['grid/nr'][()]
        ny = fl['grid/nz'][()]
        xlims = fl['grid/rlims'][()]
        ylims = fl['grid/zlims'][()]
    gmat = sparse.csr_matrix((data, indices, indptr), shape=shp)
    grid = RegularGrid(nx, ny, xlims, ylims)
    return gmat, grid


def save_dense_gmat(floc: Union[str, Path], gmat: np.ndarray, grid: RegularGrid, attrs_dct: dict = None) -> None:
    """
    Saves geometry matrix together with description of the grid.

    Currently suports only dense matrices and regular rectangular grids.

    Parameters
    ----------
    floc : str
        file location
    gmat : np.ndarray
        with shape (#chanels, #nodes)
    grid : RegularGrid
        class describing regular grid the gmat was computed for
    attrs_dict : dict, optional
        additional attributes to be saved in hdf file
    """
    floc = Path(floc)
    floc = floc.expanduser()
    if not isinstance(gmat, np.ndarray):
        msg = 'This function can only save dense gmat. ' + \
            'For saving matrix in sparse format use save_sparse_gmat function'
        raise ValueError(msg)
    with h5py.File(floc, 'w') as f:
        f.attrs['version'] = '0.1'
        f.create_dataset('data', data=gmat)
        f.attrs['format'] = 'ndarray'
        f.attrs['shape'] = gmat.shape
        grp = f.create_group('grid')
        grp.create_dataset('nr', data=grid.nr)
        grp.create_dataset('nz', data=grid.nz)
        grp.create_dataset('rlims', data=grid.rlims)
        grp.create_dataset('zlims', data=grid.zlims)
        grp.attrs['type'] = 'regular_rectangles'
        if attrs_dct is not None:
            for key in attrs_dct:
                f.attrs[key] = attrs_dct[key]
    return


def load_dense_gmat(floc: Union[str, Path]) -> Tuple[np.ndarray, RegularGrid]:
    """
    Loads hdf file and creates dense geometry matrix and RegularGrid class

    Parameters
    ----------
    floc : str
        hdf file location
    
    Returns
    -------
    gmat : np.ndarray
    grid : geometry.RegularGrid
    """
    floc = Path(floc)
    floc = floc.expanduser()
    with h5py.File(floc, 'r') as fl:
        if fl.attrs['format'] != 'ndarray':
            msg = 'This function can only load dense gmat. ' + \
                'For loading matrix in sparse format use load_sparse_gmat function'
            raise ValueError(msg)
        gmat = fl['gmat'][:]
        nx = fl['grid/nr'][()]
        ny = fl['grid/nz'][()]
        xlims = fl['grid/rlims'][()]
        ylims = fl['grid/zlims'][()]
    grid = RegularGrid(nx, ny, xlims, ylims)
    return gmat, grid
