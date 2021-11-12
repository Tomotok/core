# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Tool for easier handling of geometry matrices. Work in progress.
"""
import h5py
import warnings
from pathlib import Path


class GmatHandler(object):
    """[summary]

    Attributes
    ----------
    diag
    grid
    gmat : dict
    """

    def __init__(self, diag, grid, gmat=None):
        super().__init__()
        self.diag = diag
        self.grid = grid
        self.gmat = gmat
        rmin = grid.rmin
        rmax = grid.rmax
        zmin = grid.zmin
        zmax = grid.zmax
        nr = grid.nr
        nz = grid.nz
        self.name = 'Gmat_h5_{z}x{r}-{z1}-{z2}-{r1}-{r2}_{gt}'.format(
            r=nr, z=nz, r1=rmin, r2=rmax, z1=zmin, z2=zmax, gt='{}')

    def compute_sparse_lines(self, **kw):
        """
        Computes geometry matrix using single line of sight approx. with numerical algorithm.
        """
        xchords, ychords = self.diag.get_chord_geometry()
        from .generators import sparse_line_3d
        self.gmat = sparse_line_3d(rchord=xchords, vchord=ychords, grid=self.grid, **kw)

    def save_dense(self, gmat_path, grid=None, filename=None):
        """
        Saves geometry matrix to gmat_path. Either grid or filename must be specified.

        If filename is not specified, the default filename is generated using grid properties.

        Parameters
        ----------
        gmat_path : str or Path
            saving folder
        grid : Pixgrid, optional
        filename : str, optional
        """
        gmat_path = Path(gmat_path)
        try:
            assert grid is not None or filename is not None
        except AssertionError:
            raise ValueError('Either grid or filename must be specified.')
        if filename is None:
            gpath = Path(gmat_path) / self.name.format(self.gmat['type'])
        else:
            gpath = Path(gmat_path) / filename
        try:
            gpath.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(str(gpath), 'w') as hdfgmat:
                for key in ['values', 'z', 'r']:
                    hdfgmat[key] = self.gmat[key]
                hdfgmat.attrs['type'] = self.gmat['type']
        except OSError as e:
            warnings.warn(
                'Can not create or write to gmat_path. Geometry matrix was not saved. {}'.format(e))

    def load_dense(self, gpath, filename=None):
        """
        Loads geometry matrix.

        Assumes that gpath is full path to gmat file if filename is omitted.

        Parameters
        ----------
        gpath : str or Path
        filename : str, optional
            name of file containing geometry matrix

        Returns
        -------
        dict
        """
        gpath = Path(gpath)
        if self.gmat is None:
            self.gmat = {}
        if filename is not None:
            gpath = gpath / filename
        with h5py.File(str(gpath), 'r') as hdfgmat:
            for key in ['values', 'z', 'r']:
                self.gmat[key] = hdfgmat[key][...]
            self.gmat['type'] = hdfgmat.attrs['type']

    def fetch(self, gmat_path='', gtype='line', save=False, force_comp=False):
        """
        Tries to load geometry matrix for specified grid from gmat path.
        Computes new matrix if appropriate matrix is not available.

        Parameters
        ----------
        gmat_path : str or Path, optional
        gtype : str, optional
            Specifies type of geometry matrix to be used. {'line', 'poly'} are supported for computation.
        save : bool, optional
        force_comp : bool, optional
            Forces computation of geometry matrix
        """
        grid = self.grid
        name = self.name.format(gtype)
        gpath = Path(gmat_path) / name
        try:
            assert force_comp is False
            self.load(gpath)
        except (OSError, AssertionError):
            print('Unable to load geom. matrix from {}.'.format(gpath),
                  'Generating new geometry matrix.')
            if gtype == 'line':
                self.compute_lines()
            elif gtype == 'poly':
                self.compute_polygons()
            if save:
                self.save(gmat_path, grid)
