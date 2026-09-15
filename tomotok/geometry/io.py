# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Saving and loading functions for geometry matrices in sparse format using h5py module.
"""
from pathlib import Path
import h5py
import numpy as np
import scipy.sparse as sparse

from tomotok.tools.hdf import sparse_to_hdf, hdf_to_sparse
from .grids import RegularGrid


def save_sparse_gmat(floc: str | Path, gmat: sparse.sparray, grid: RegularGrid, attrs_dct: dict=None) -> None:
    """
    Saves geometry matrix together with description of the grid.

    Currently supports only regular rectangular grids.

    Parameters
    ----------
    floc : str or pathlib.Path
        file location
    gmat : sparse.sparray
        with shape (#chanels, #nodes)
    grid : RegularGrid
        class describing regular grid the gmat was computed for
    attrs_dict : dict, optional
        additional attributes to be saved in hdf file
    """
    floc = Path(floc)
    floc = floc.expanduser()
    if not isinstance(gmat, sparse.sparray):
        msg = 'This function can only save sparse geometry matrix.'
        raise ValueError(msg)
    if not isinstance(grid, RegularGrid):
        msg = 'This function can only save geometry matrices computed for regular grids.'
        raise ValueError(msg)
    with h5py.File(floc, 'w') as f:
        f.attrs['version'] = '0.2'
        gmg = f.create_group('gmat')
        sparse_to_hdf(gmat, gmg)
        grp = f.create_group('grid')
        grp.create_dataset('nr', data=grid.nr)
        grp.create_dataset('nz', data=grid.nz)
        grp.create_dataset('rlims', data=grid.rlims)
        grp.create_dataset('zlims', data=grid.zlims)
        grp.attrs['type'] = 'regular_rectangles'
        if attrs_dct is not None:
            for key in attrs_dct:
                f.attrs[key] = attrs_dct[key]


def load_sparse_gmat(floc: str | Path) -> tuple[sparse.sparray, RegularGrid]:
    """
    Loads hdf file and creates geometry matrix in sparse array format and RegularGrid class

    Parameters
    ----------
    floc : str or pathlib.Path
        hdf file location
    
    Returns
    -------
    gmat : sparse.sparray
    grid : geometry.RegularGrid
    """
    floc = Path(floc)
    floc = floc.expanduser()
    with h5py.File(floc, 'r') as fl:
        if fl.attrs['version'] == '0.1':
            data = fl['data'][:]
            indptr = fl['indptr'][:]
            indices = fl['indices'][:]
            shp = fl.attrs['shape']
            nr = fl['grid/nr'][()]
            nz = fl['grid/nz'][()]
            rlims = fl['grid/rlims'][()]
            zlims = fl['grid/zlims'][()]
            gmat = sparse.csr_array((data, indices, indptr), shape=shp)
        elif fl.attrs['version'] == '0.2':
            gmat = hdf_to_sparse(fl['gmat'])
            if fl['grid'].attrs['type'] != 'regular_rectangles':
                raise ValueError(
                    f'Unsupported grid type: {fl["grid"].attrs["type"]}.'+
                    'This function can only load regular rectangular grids.'
                )
            nr = fl['grid/nr'][()]
            nz = fl['grid/nz'][()]
            rlims = fl['grid/rlims'][()]
            zlims = fl['grid/zlims'][()]
        else:
            raise ValueError(f'Unsupported file version: {fl.attrs["version"]}')
    grid = RegularGrid(nr, nz, rlims, zlims)
    return gmat, grid


def save_dense_gmat(floc: str | Path, gmat: np.ndarray, grid: RegularGrid, attrs_dct: dict = None) -> None:
    """
    Saves geometry matrix together with description of the grid.

    Currently suports only dense matrices and regular rectangular grids.

    Parameters
    ----------
    floc : str or pathlib.Path
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


def load_dense_gmat(floc: str | Path) -> tuple[np.ndarray, RegularGrid]:
    """
    Loads hdf file and creates dense geometry matrix and RegularGrid class

    Parameters
    ----------
    floc : str or pathlib.Path
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
        gmat = fl['data'][:]
        nr = fl['grid/nr'][()]
        nz = fl['grid/nz'][()]
        rlims = fl['grid/rlims'][()]
        zlims = fl['grid/zlims'][()]
    grid = RegularGrid(nr, nz, rlims, zlims)
    return gmat, grid
