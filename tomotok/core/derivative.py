# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Handles computation of derivative matrices used for regularization in MFR algorithm.
"""
from warnings import warn

import numpy as np
from scipy import sparse
from scipy.sparse import spdiags, eye

from .geometry import RegularGrid




def prepare_mag_data(flux):
    """
    Calculates gradient map of magnetic flux function Psi(R,z) and returns
    arcus tangens of dPsi_y/-dPsi_x that is a suitable funciton for anizotropic
    diffusion matrix calculation

    Parameters
    ----------
    flux : numpy.ndarray
        array of Psi(R,z) evolution, with axes (z, R, t)

    Returns
    ------
    numpy.ndarray
        arcus tangens of Psi gradient
    """
    vgrad = np.gradient(flux[:, :])
    atan2 = np.arctan2(vgrad[1], -vgrad[0])
    return atan2


def px_norm(direc):
    """
    Computed (normalised?) distance between centers of nodes.
    """
    # FIXME: assumes square grid
    dn = np.sqrt(2)
    mod = np.asarray([s % 2 for s in direc])
    norms = np.zeros(np.shape(direc))
    norms[mod == 0] = 1
    norms[mod == 1] = dn
    return norms


def generate_anizo_matrix(grid, atan2, derivative):
    """
    Write prepared directions atan2 into the derivative matrix.
    Main magic of this algorithm. Rewrite arctan into the directions
    and decompose directions to parallel and oblique direction.

    Parameters
    ----------
    grid : RegularGrid
        Object with pixel grid gridinates
    atan2 : numpy.ndarray
        3D array of arcus tangents of Psi(R,z) evolution
    derivative : int
        derivative type identificator

    Returns
    -------
    bper : scipy.sparse
        sparse matrix with perpendicular derivatives
    bpar : scipy.sparse
        sparse matrix with parallel derivatives
    bpar_tmp: numpy.array
        (optional) dense matrix with parallel derivatives
    bper_tmp: numpy.array
        (optional) dense matrix with perpendicular derivatives
    """
    # obtain pixel dimensions
    nx = grid.nr
    ny = grid.nz
    npix = grid.nodes_num

    atan2 = atan2.flatten()

    # initiate 9point derivative matrices
    bper_tmp = np.zeros((npix, 9))
    bpar_tmp = np.zeros((npix, 9))
    center_tmp = 4

    # check for nans, potentially useful if vessel/separatrix truncated is supplied
    ind = ~np.isnan(atan2)
    atan2 = atan2[ind]
    n_ind = len(atan2)

    # decomposition of flux contour direction to two neighboring pixels
    for k in [0, 1]:
        # pixel with maximum contribution and px with second maximum contribution (45deg neighbour)
        direction = np.int_(np.mod(np.floor(atan2 / (np.pi/4) + k), 8))

        # array of reference indices
        dir_ = np.array((-1, 2, 3, 4, 1, -2, -3, -4), dtype=int)  # F like
        # dir_ = np.array((-3, -2, 1, 4, 3, 2, -1, -4), dtype=int)  # C like
        # C  [-4, -3, -2]       F  [-4, -1,  2]
        #    [-1,  0,  1]          [-3,  0,  3]
        #    [ 2,  3,  4]          [-2,  1,  4]
        # dir_ = np.array((-1, ny-1, ny, ny+1, 1, -ny+1, -ny, -ny-1), dtype=int)  # is compressed form

        next_ = np.squeeze(dir_[direction])

        # saw function, MAIN MAGIC, first steps to find projections of
        # the direction to the two neighboring pixels

        arelativ = np.abs(np.pi/4 - np.mod(atan2 + np.pi/4, np.pi/2))

        # obligue direction, zoom => ugly hack (works :)
        k1 = np.sin(2*arelativ)
        # direction paralel with axes
        k2 = np.cos(2*arelativ)

        a = np.zeros(n_ind)
        ind_mod2 = np.bool_(np.mod(direction, 2))
        a[ind_mod2] = k1[ind_mod2]
        a[~ind_mod2] = k2[~ind_mod2]
        forind = center_tmp + next_
        backind = center_tmp - next_

        # Assign a value to specific pixels depending on the desired difference scheme:
        # 1 and 2 are just depending on the axis direction, 3 is second derivative and
        # 4 is central derivative
        # normalise pixels

        if derivative == 1:
            bper_tmp[ind, forind] = a/px_norm(forind)
        elif derivative == 2:
            bper_tmp[ind, backind] = a/px_norm(backind)
        elif derivative == 3:
            bper_tmp[ind, forind] = a/px_norm(forind)
            bper_tmp[ind, backind] = a/px_norm(backind)
        elif derivative == 4:
            bper_tmp[ind, forind] = a/px_norm(forind)
            bper_tmp[ind, backind] = -a/px_norm(backind)
        else:
            raise ValueError("Bad derivative type number, allowed {1,2,3,4}")

    # Constructing matrix with directions parallel to the magnetic field
    # from the matrix with directions perpendicular to magnetic field by essentially rotating
    # the outer pixels by 90deg
    ind = np.arange(8)
    pt1 = dir_[np.mod(ind, 8)]
    pt2 = dir_[np.mod(ind+2, 8)]
    # rotate directions by 90deg
    bpar_tmp[:, center_tmp + pt2[ind]] = bper_tmp[:, center_tmp + pt1[ind]]

    # normalisation and treatment for the central pixel: -1,
    # except for central derivative where it is zero

    if derivative in (1, 2, 3):
        bper_tmp = sparse.spdiags(1 / (np.sum(bper_tmp, 1) + 0.000001), 0, npix, npix
                                  ) * bper_tmp
        bpar_tmp = sparse.spdiags(1 / (np.sum(bpar_tmp, 1) + 0.000001), 0, npix, npix
                                  ) * bpar_tmp
        bper_tmp[:, center_tmp] = -1
        bpar_tmp[:, center_tmp] = -1
    elif derivative == 4:
        bper_tmp = sparse.spdiags(1 / (np.sum(np.abs(bper_tmp), 1)), 0, npix, npix
                                  ) * bper_tmp
        bpar_tmp = sparse.spdiags(1 / (np.sum(np.abs(bpar_tmp), 1)), 0, npix, npix
                                  ) * bpar_tmp
    else:
        raise ValueError("Bad derivative type number, allowed {1,2,3,4}")
    # final conversion to npix x npix diagonal sparse matrices used in the calculation
    # bpar = sparse.spdiags(bpar_tmp.T, (ny+1, ny, ny-1, 1, 0, -1, -ny+1, -ny, -ny-1), npix, npix).T
    # bper = sparse.spdiags(bper_tmp.T, (ny+1, ny, ny-1, 1, 0, -1, -ny+1, -ny, -ny-1), npix, npix).T
    bpar = sparse.spdiags(bpar_tmp.T, (nx + 1, 1, -nx + 1, nx, 0, -nx, nx - 1, -1, -nx - 1), npix, npix).T
    bper = sparse.spdiags(bper_tmp.T, (nx + 1, 1, -nx + 1, nx, 0, -nx, nx - 1, -1, -nx - 1), npix, npix).T
    bpar = sparse.csc_matrix(bpar)
    bper = sparse.csc_matrix(bper)
    return bpar, bper, bpar_tmp, bper_tmp


def reduce_matrix(mat, mask, compensate_edges=True):
    """
    Creates reduced matrix by cutting out rows and columns representing unwanted nodes.

    Parameters
    ----------
    mat : scipy.sparse.csr_matrix
        matrix to be reduced
    mask : numpy.ndarray of bool
        array or mask array to select desired nodes

    Returns
    -------
    scipy.sparse.csr_matrix
        reduced matrix
    """
    if mask.ndim == 2:
        mask = mask.flatten()
    elif mask.ndim > 2:
        raise ValueError('Mask must be 1D or 2D array.')
    mat = mat[mask, :][:, mask]
    if compensate_edges:
        row_sum = np.array(mat.sum(1)).flatten()
        row_sum_diag = sparse.diags([row_sum], [0], format='csr')
        mat = mat - row_sum_diag
    return mat


def derivative_matrix(grid: RegularGrid, direction: str, scheme: str = 'forward', mask: np.ndarray = None, compensate_edges=True):
    """
    Creates a derivative matrix using numerical differences

    Parameters
    ----------
    grid : RegularGrid
    direction : str
        Determines one of the 8 possible directions that can be used.
        Specified using the over edge neighbor 'right', 'top', 'left', 'bottom' or
        combination over corner directions like 'top-left'
    scheme : str, optional
        Selects numerical scheme to be used. Can be 'forward', 'backward', 'central', 'second'.
        The default value is 'forward'.
    mask : numpy.ndarray, optional
        A bool mask determining nodes of regular grid to keep, by default None.
        Rows and columns representing False nodes are removed from derivative matrix.
        If mask is None, all rows and columns are returned.
    compensate_edges : bool, optional
        Subtracts from diagonal so that sum of each row is zero, by default False.

    Returns
    -------
    scipy.sparse.csc_matrix
    """
    center = sparse.diags([1], [0], shape=(grid.size, grid.size), format='csc')
    right = sparse.diags([1], [1], shape=(grid.size, grid.size), format='csc')
    upper_right = sparse.diags([1], [grid.nr + 1], shape=(grid.size, grid.size), format='csc')
    upper = sparse.diags([1], [grid.nr], shape=(grid.size, grid.size), format='csc')
    upper_left = sparse.diags([1], [grid.nr - 1], shape=(grid.size, grid.size), format='csc')
    left = sparse.diags([1], [-1], shape=(grid.size, grid.size), format='csc')
    lower_left = upper = sparse.diags([1], [-grid.nr - 1], shape=(grid.size, grid.size), format='csc')
    lower = sparse.diags([1], [-grid.nr], shape=(grid.size, grid.size), format='csc')
    lower_right = upper = sparse.diags([1], [-grid.nr + 1], shape=(grid.size, grid.size), format='csc')

    if direction == 'right':
        following = right
        previous = left
        step = grid.dr
    elif direction == 'top-right':
        following = upper_right
        previous = lower_left
        step = (grid.dr*2 + grid.dz**2)**0.5
    elif direction == 'top':
        following = upper
        previous = lower
        step = grid.dz
    elif direction == 'top-left':
        following = upper_left
        previous = lower_right
        step = (grid.dr*2 + grid.dz**2)**0.5
    elif direction == 'left':
        following = left
        previous = right
        step = grid.dr
    elif direction == 'bottom-left':
        following = lower_left
        previous = upper_right
        step = (grid.dr*2 + grid.dz**2)**0.5
    elif direction == 'bottom':
        following = lower
        previous = upper
        step = grid.dz
    elif direction == 'bottom-right':
        following = lower_right
        previous = upper_left
        step = (grid.dr*2 + grid.dz**2)**0.5
    else:
        raise ValueError(f'Unknown direction {direction}.')

    if scheme == 'forward':
        dmat = following - center
    elif scheme == 'backward':
        dmat = center - previous
    elif scheme == 'central':
        dmat = following - previous
        dmat /= 2
    elif scheme == 'second':
        dmat = following - 2 * center + previous
        dmat = dmat / step
    else:
        schemes = ['forward', 'backward', 'central', 'second']
        msg = 'Uknown numerical derivatie scheme {}. Use one of {}.'
        raise ValueError(msg.format(scheme, schemes))

    # normalisation
    dmat = dmat / step
    if mask is not None:
        dmat = reduce_matrix(dmat, mask, compensate_edges)
    return dmat


def laplace_matrix(grid: RegularGrid, mask=None, compensate_edges=True, diagonals=True):
    """
    Creates sparse laplace matrix.

    Parameters
    ----------
    grid : RegularGrid
    mask : numpy.ndarray, optional
        A bool mask determining nodes of regular grid to keep, by default None.
        Rows and columns representing False nodes are removed from derivative matrix.
        If mask is None, all rows and columns are returned.
    compensate_edges : bool, optional
        Subtracts from diagonal so that sum of each row is zero, by default False.
    diagonals : bool, optional
        Selects whether to use diagonal neighbors in matrix, by default True.

    Returns
    -------
    scipy.sparse.csc_matrix
    """
    center = sparse.diags([1], [0], shape=(grid.size, grid.size), format='csc')
    right = sparse.diags([1], [1], shape=(grid.size, grid.size), format='csc')
    upper = sparse.diags([1], [grid.nr], shape=(grid.size, grid.size), format='csc')
    left = sparse.diags([1], [-1], shape=(grid.size, grid.size), format='csc')
    lower = sparse.diags([1], [-grid.nr], shape=(grid.size, grid.size), format='csc')

    lmat = (right + left) / grid.dr
    lmat += (upper + lower) / grid.dz
    lmat -= (2 / grid.dr + 2 / grid.dz) * center

    if diagonals:
        upper_right = sparse.diags([1], [grid.nr + 1], shape=(grid.size, grid.size), format='csc')
        upper_left = sparse.diags([1], [grid.nr - 1], shape=(grid.size, grid.size), format='csc')
        lower_left = sparse.diags([1], [-grid.nr - 1], shape=(grid.size, grid.size), format='csc')
        lower_right = sparse.diags([1], [-grid.nr + 1], shape=(grid.size, grid.size), format='csc')
        diagonal_distance = (grid.dr**2 + grid.dz**2)**0.5

        lmat += (upper_right + upper_left + lower_left + lower_right) / diagonal_distance
        lmat -= 4 / diagonal_distance * center

    if mask is not None:
        lmat = reduce_matrix(lmat, mask, compensate_edges)
    return lmat


def anisotropic_derivative_matrices(grid: RegularGrid, magnetic_flux, mask=None, compensate_edges=True):
    """
    Computes anisotropic derivative matrices.

    Uses forward and backward derivative schemes for both parallel and perpendicular directions.

    Parameters
    ----------
    grid : RegularGrid
        A reconstruction grid
    magnetic_flux : numpy.ndarray
        values of psi normalized interpolated to grid
    mask : numpy.ndarray, optional
        bool mask
    compensate_edges : bool, optional
        if True, subtracts row sum from diagonal, by default False

    Returns
    -------
    list of scipy.sparse.csrmatrix
        list of anistropic derivative matrices
        [parallel forward, perpendicular forward, parallel backward, perpendicular backward]
    """
    vgrad = np.gradient(magnetic_flux[:, :])
    atan2 = np.arctan2(vgrad[1], -vgrad[0])
    bpar1, bper1, _, _ = generate_anizo_matrix(grid, atan2, 1)  # forward
    bpar2, bper2, _, _ = generate_anizo_matrix(grid, atan2, 2)  # backward

    dmat_list = [bpar1, bper1, bpar2, bper2]
    
    if mask is not None:
        for i, dmat in enumerate(dmat_list):
            reduced = reduce_matrix(dmat, mask, compensate_edges)
            dmat_list[i] = reduced

    return dmat_list
