# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Module with numeric generators of geometry matrix.
"""
import time
from warnings import warn

import numpy as np
import scipy.sparse as sparse
from numpy.typing import ArrayLike

from .grids import RegularGrid


def sparse_line_3d(rchord: ArrayLike, vchord: ArrayLike, grid: RegularGrid, 
                   ychord: ArrayLike = None, step=1e-3, rmin: float = None) -> sparse.csr_matrix:
    """
    Computes geometry matrix using simple numerical integration algorithm.

    Assumes toroidally symmetric reconstruction nodes. 
    Optimized version working with sparse matrices.

    .. deprecated:: 2.0
        Use :func:`sparse_line` instead.

    Parameters
    ----------
    rchord : array-like
        radial coordinates of chord start and end points
    vchord : array-like
        vertical coordinates of chord start and end points
    grid : RegularGrid
    ychord : array_like, optional
        if specified, 3D Cartesian coordinates are assumed and rchord represents coordinates of x axis
    step : float, optional
        Integration step in meters.
    rmin : float, optional
        Stop iteration if r is lower. Useful when chords may go through central column.

    Returns
    -------
    sparse.csr_matrix

    See Also
    --------
    sparse_line
    """
    warn('Deprecated by `sparse_line`', DeprecationWarning)
    if ychord is None:
        ychord = np.zeros_like(rchord)
    rchord = np.array(rchord, ndmin=2)
    vchord = np.array(vchord, ndmin=2)
    ychord = np.array(ychord, ndmin=2)
    ela = time.time()
    nch = rchord.shape[0]
    gmat = sparse.lil_matrix((nch, grid.size))
    dr = np.diff(rchord, axis=1)
    dv = np.diff(vchord, axis=1)
    dy = np.diff(ychord, axis=1)
    dst = np.sqrt(dr * dr + dy * dy + dv * dv)
    for i in range(nch):
        steps = int(dst[i] / step)
        x = np.linspace(rchord[i, 0], rchord[i, 1], steps)
        y = np.linspace(ychord[i, 0], ychord[i, 1], steps)
        z = np.linspace(vchord[i, 0], vchord[i, 1], steps)
        r = np.sqrt(x**2 + y**2)
        if rmin is not None:
            hit = np.any(r < rmin)
            if hit:
                idx = r.argmin()
                r = r[:idx]
                z = z[:idx]
        hist = np.histogram2d(r, z, bins=[grid.r_border, grid.z_border])
        row = hist[0].T * dst[i] / steps
        srow = sparse.coo_matrix(row.flatten())
        gmat[i] = srow
    ela = time.time() - ela
    print('Gmat generation time {:.2f}s'.format(ela),
          'average time per chord {:.0f}ms'.format(ela / nch * 1000))
    return gmat.tocsr()


def calcam_sparse_line_3d(pupil: ArrayLike, dirs: ArrayLike, grid: RegularGrid, 
                          step=1e-3, rmin: float = None, elong=1., steps: float = None) -> sparse.csr_matrix:
    """
    Computes geometry matrix from calcam input using sparse_line_3d algorithm.

    Assumes that pupil and dirs coordinates are (x, y, z) = (horizontal, horizontal, vertical)

    .. deprecated:: 2.0
        Use :func:`calcam_sparse_line` instead.

    Parameters
    ----------
    pupil : np.ndarray
        (x, y, z) coordinates of pupil position
    dirs : np.ndarray
        (#rows, #columns, 3) line of sight direction vector coordinates
    grid : tomotok.core.geometry.RegularGrid
        reconstruction grid
    step : float, optional
        Integration step in meters.
    rmin : float, optional
        cut lines of sight if they intersect cylinder with radius of rmin, by default 0.3
    elong : float, optional
        multiplier for direction vectors elongation

    Returns
    -------
    sparse.csr_matrix

    See Also
    --------
    calcam_sparse_line
    """
    warn('Deprecated by `calcam_sparse_line`', DeprecationWarning)
    if steps is not None:
        warn('"steps" parameter was deprecated by "step"', DeprecationWarning)
        step = steps
    if elong != 1.:
        dirs = elong * dirs
    dirs = dirs.reshape(-1, 3)
    xchords = np.ones((dirs.shape[0], 2)) * pupil[0]
    xchords[:, 1] += dirs[:, 0]
    ychords = np.ones((dirs.shape[0], 2)) * pupil[1]
    ychords[:, 1] += dirs[:, 1]
    zchords = np.ones((dirs.shape[0], 2)) * pupil[2]
    zchords[:, 1] += dirs[:, 2]
    gmat = sparse_line_3d(xchords, zchords, grid, ychords, step, rmin)
    return gmat


def sparse_line(starts: np.ndarray, ends: np.ndarray, grid: RegularGrid, step: float = 1e-3, rmin: float = 0):
    """
    Computes geometry matrix using simple numerical integration algorithm.

    Uses lines of sight start and end points in 3D Cartesian coordinates as input.

    Parameters
    ----------
    starts, ends : ndarray
        Contains lines of sight start/end points, with shape (..., 3)
    grid : RegularGrid
        Reconstruction grid
    step : float, optional
        Integration step, by default 1e-3
    rmin : float, optional
        Stops the lines of sight if they intersect cylinder with radius of rmin, by default -1

    Returns
    -------
    sparse.csr_matrix
    """
    if starts.ndim > 2:
        try:
            starts = starts.reshape(-1, 3)
            ends = ends.reshape(-1, 3)
        except ValueError:
            raise ValueError('starts and ends must be ndarrays with shape (..., 3)')
    diff = ends - starts
    dst = np.linalg.norm(diff, axis=1)
    line_num = diff.shape[0]
    gmat = sparse.lil_matrix((line_num, grid.size))
    for i in range(line_num):
        steps = int(dst[i] / step)
        x = np.linspace(starts[i, 0], ends[i, 0], steps)
        y = np.linspace(starts[i, 1], ends[i, 1], steps)
        z = np.linspace(starts[i, 2], ends[i, 2], steps)
        r = np.sqrt(x**2 + y**2)
        if rmin > 0:
            hit = np.any(r < rmin)
            if hit:
                idx = r.argmin()
                r = r[:idx]
                z = z[:idx]
        hist = np.histogram2d(r, z, bins=[grid.r_border, grid.z_border])
        row = hist[0].T * dst[i] / steps
        srow = sparse.coo_matrix(row.flatten())
        gmat[i] = srow
    return gmat.tocsr()


def calcam_sparse_line(pupil: np.ndarray, endpoints: np.ndarray, grid: RegularGrid, **kw):
    """Wrapper for typical calcam input.
    
    Parameters
    ----------
    pupil : np.ndarray
        (x, y, z) coordinates of pupil position
    endpoints : np.ndarray
        (#rows, #columns, 3) line of sight end point coordinates
    grid : tomotok.core.geometry.RegularGrid
        reconstruction grid
    **kw : dict
        additional parameters passed to sparse_line

    See Also
    --------
    sparse_line
    """
    startpoints = np.ones_like(endpoints) * pupil
    return sparse_line(startpoints, endpoints, grid, **kw)


def dense_line(starts: np.array, ends: np.array, grid: RegularGrid, step: float = 1e-3, rmin: float = 0):
    """
    Computes geometry matrix using simple numerical integration algorithm.

    Uses lines of sight start and end points in 3D Cartesian coordinates as input.

    Parameters
    ----------
    starts, ends : ndarray
        Contains lines of sight start/end points, with shape (..., 3)
    grid : RegularGrid
        Reconstruction grid
    step : float, optional
        Integration step, by default 1e-3
    rmin : float, optional
        Stops the lines of sight if they intersect cylinder with radius of rmin, by default 0
    
    Returns
    -------
    ndarray
        geometry matrix with shape (..., #nodes)
    """
    if starts.ndim > 2:
        try:
            starts = starts.reshape(-1, 3)
            ends = ends.reshape(-1, 3)
        except ValueError:
            raise ValueError('starts and ends must be ndarrays with shape (..., 3)')
    diff = ends - starts
    dst = np.linalg.norm(diff, axis=1)
    line_num = diff.shape[0]
    gmat = np.zeros((line_num, grid.size))
    for i in range(line_num):
        steps = int(dst[i] / step)
        x = np.linspace(starts[i, 0], ends[i, 0], steps)
        y = np.linspace(starts[i, 1], ends[i, 1], steps)
        z = np.linspace(starts[i, 2], ends[i, 2], steps)
        r = np.sqrt(x**2 + y**2)
        if rmin > 0:
            hit = np.any(r < rmin)
            if hit:
                idx = r.argmin()
                r = r[:idx]
                z = z[:idx]
        hist = np.histogram2d(r, z, bins=[grid.r_border, grid.z_border])
        row = hist[0].T * dst[i] / steps
        gmat[i] = row.flatten()
    return gmat
