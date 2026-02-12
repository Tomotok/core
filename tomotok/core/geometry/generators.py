# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Module with numeric generators of geometry matrix.
"""
import numpy as np
import scipy.sparse as sparse

from .grids import RegularGrid


def sparse_line(
        starts: np.ndarray, ends: np.ndarray, grid: RegularGrid, step: float = 1e-3, rmin: float = -1
) -> sparse.csr_array:
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
    gmat = sparse.lil_array((line_num, grid.size))
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
        srow = sparse.coo_array(row.flatten())
        gmat[i] = srow
    return gmat.tocsr()
