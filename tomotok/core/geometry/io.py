# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Saving and loading functions for geometry matrices in sparse format using h5py module.
"""
from typing import Tuple
import h5py

import scipy.sparse as sparse

from .grids import RegularGrid


def save_sparse_gmat(floc, gmat: sparse.csr_matrix, grid: RegularGrid, attrs_dct: dict=None) -> None:
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
    if not isinstance(gmat, sparse.csr_matrix):
        msg = 'This function can only save sparse gmat. ' + \
            'For saving matrix in dense format use core.io.to_hdf function'
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


def load_sparse_gmat(floc: str) -> Tuple[sparse.csr_matrix, RegularGrid]:
    """
    Loads hdf file and creates sparse csr_matrix and RegularGrid class

    Parameters
    ----------
    floc : str
        hdf file location
    
    Returns
    -------
    gmat : sparse.csr_matrix
    grid : geometry.RegularGrid
    """
    with h5py.File(floc, 'r') as fl:
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
