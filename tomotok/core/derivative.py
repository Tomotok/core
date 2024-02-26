# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Handles computation of derivative matrices used for regularization in MFR algorithm.
"""
from typing import List, Optional
import numpy as np
from scipy import sparse

from .geometry import RegularGrid


def all_direction_derivative_matrices(
        grid: RegularGrid, scheme: str = 'forward', mask: Optional[np.ndarray] = None, compensate_edges=True
    ) -> List[sparse.csc_matrix]:
    """
    Creates derivative matrices for all 8 directions using provided scheme.

    Parameters
    ----------
    grid : RegularGrid
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
    list of scipy.sparse.csc_matrix
        List of derivative matrices for all 8 directions with following order:
        right, top-right, top, top-left, left, bottom-left, bottom, bottom-right
    """
    derivatives = [
        derivative_matrix(grid, 'right', scheme, mask, compensate_edges),
        derivative_matrix(grid, 'top-right', scheme, mask, compensate_edges),
        derivative_matrix(grid, 'top', scheme, mask, compensate_edges),
        derivative_matrix(grid, 'top-left', scheme, mask, compensate_edges),
        derivative_matrix(grid, 'left', scheme, mask, compensate_edges),
        derivative_matrix(grid, 'bottom-left', scheme, mask, compensate_edges),
        derivative_matrix(grid, 'bottom', scheme, mask, compensate_edges),
        derivative_matrix(grid, 'bottom-right', scheme, mask, compensate_edges),
    ]
    return derivatives


def standard_anisotropic_derivative_matrices(
        grid: RegularGrid, flux: np.ndarray, mask: Optional[np.ndarray] = None, compensate_edges=True
    ) -> List[sparse.csc_matrix]:
    """
    Creates isotropic derivative matrices for parallel and perpendicular directions.

    Parameters
    ----------
    grid : RegularGrid
    fluxes : numpy.ndarray
        Matrix with magnetic flux values, shape has to match grid
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
    list of scipy.sparse.csc_matrix
        List of derivative matrices for parallel and perpendicular directions in following order:
        parallel counter clockwise, perpendicular counter clockwise, parallel clockwise, perpendicular clockwise
    """
    derivatives = [
        anisotropic_derivative_matrix(grid, flux, 'parallel', 'forward', mask, compensate_edges),
        anisotropic_derivative_matrix(grid, flux, 'perpendicular', 'forward', mask, compensate_edges),
        anisotropic_derivative_matrix(grid, flux, 'parallel', 'backward', mask, compensate_edges),
        anisotropic_derivative_matrix(grid, flux, 'perpendicular', 'backward', mask, compensate_edges),
    ]
    return derivatives


def reduce_matrix(mat: sparse.spmatrix, mask: np.ndarray) -> sparse.spmatrix:
    """
    Creates reduced matrix by cutting out rows and columns representing unwanted nodes.

    Parameters
    ----------
    mat : scipy.sparse.spmatrix
        matrix to be reduced
    mask : numpy.ndarray of bool
        mask array for selecting desired nodes, True for nodes to keep
        must be 1D or 2D, size must match grid size (# rows or # columns)

    Returns
    -------
    scipy.sparse.csc_matrix
        reduced matrix
    """
    if mask.ndim == 2:
        mask = mask.flatten()
    elif mask.ndim > 2:
        raise ValueError('Mask must be 1D or 2D array.')
    if mask.size != mat.shape[0]:
        raise ValueError('Mask size does not match derivative matrix size.')
    mat = mat[mask, :][:, mask]
    return mat


def compensate_matrix(mat: sparse.spmatrix) -> sparse.spmatrix:
    """
    Subtracts from diagonal so that sum of each row is zero.

    Parameters
    ----------
    mat : scipy.sparse.spmatrix
        matrix to be compensated

    Returns
    -------
    scipy.sparse.csc_matrix
        compensated matrix
    """
    row_sum = np.array(mat.sum(1)).flatten()
    row_sum_diag = sparse.diags([row_sum], [0], format='csc')
    mat = mat - row_sum_diag
    return mat


def derivative_matrix(
        grid: RegularGrid, direction: str, scheme: str = 'forward', 
        mask: Optional[np.ndarray] = None, compensate_edges=True
    ) -> sparse.csc_matrix:
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
        dmat = reduce_matrix(dmat, mask)
    if compensate_edges:
        dmat = compensate_matrix(dmat)
    return dmat


def laplace_matrix(
        grid: RegularGrid, 
        mask: Optional[np.ndarray] = None, compensate_edges: bool = True, diagonals: bool = True
    ) -> sparse.csc_matrix:
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
        lmat = reduce_matrix(lmat, mask)
    if compensate_edges:
        lmat = compensate_matrix(lmat)
    return lmat


def anisotropic_derivative_matrix(
        grid: RegularGrid, fluxes: np.ndarray, direction: str = 'parallel', scheme: str = 'forward',
        mask: Optional[np.ndarray] = None, compensate_edges: bool = True
    ) -> sparse.csc_matrix:
    """
    Computes derivative matrix with varying direction based on flux surfaces shapes.

    Uses direction of gradient to distribute contribution to two neighboring nodes.
    These are designated as previous and next as the direction rotates.
    The weight of next node is equal to remnant of direction angle division by 1.
    The weight of previous node is equal to 1 minus the weight of next node.

    Parameters
    ----------
    grid : RegularGrid
        Reconstruction grid definition
    fluxes : numpy.ndarray
        Matrix with magnetic flux values, shape has to match grid
    mask : numpy.ndarray of bool, optional
        _description_, by default None
    direction : str, optional
        _description_, by default 'parallel'
    scheme : str, optional
        derivative scheme used, by default 'forward'
         - forward is in the direction of positive angle i.e. counter clockwise
         - backward is in the direction of negative angle i.e. clockwise

    Returns
    -------
    scipy.sparse.csc_matrix
        _description_
    """
    if grid.shape != fluxes.shape:
        raise ValueError('Grid shape does not match fluxes shape.')
    if scheme not in ['forward', 'backward']:
        raise ValueError('Scheme must be either `forward` or `backward`.')
    if direction not in ['parallel', 'perpendicular']:
        raise ValueError('Direction must be either `parallel` or `perpendicular`.')

    grad_z, grad_r = np.gradient(fluxes, grid.z_center, grid.r_center)
    atan2 = np.arctan2(grad_z, grad_r)  # direction angle of flux surface gradient in radians
    atan_mod = (atan2.flatten() - atan2.min()) / (2 * np.pi) * 8  # rescale atan2 to <0, 8) range
    # atan_mod = atan_mod.flatten() # % 8  # FIXME: remove?

    rotation = 0
    if direction == 'parallel':
        rotation += 2  # rotate gradient by 90 degrees (2 * pi/4) to get tangent
    if scheme=='backward':
        rotation += 4  # rotate directions by 180 degrees (4 * pi/4)

    directions_prev = (atan_mod + rotation) % 8
    directions_prev = np.floor(directions_prev)
    directions_next = (directions_prev + 1) % 8

    weight_next = (atan_mod % 1).flatten()
    weight_prev = 1 - weight_next

    # array to store diagonals passed to sparse constructor
    diagonals = np.zeros((grid.size, 9))
    for i in range(8):
        mask_prev = directions_prev == i
        mask_next = directions_next == i
        diagonals[mask_prev, i] += weight_prev[mask_prev]
        diagonals[mask_next, i] += weight_next[mask_next]

    # normalize by center distance
    horizontal = grid.dr
    vertical = grid.dz
    diagonal = (horizontal**2 + vertical**2)**0.5
    diagonals[:, [1, 3, 5, 7]] /= diagonal  # lb, rb, lu, ru
    diagonals[:, [0, 4]] /= horizontal  # l, r
    diagonals[:, [2, 6]] /= vertical  # b, u
    # center node
    sum_node = diagonals.sum(axis=1)
    diagonals[:, 8] = - sum_node

    # order of directions from angle to relative index to central pixel
    # left, lower left, lower, lower right, right, upper right, upper, upper left, central
    offsets = (-1, -grid.nr -1, -grid.nr, -grid.nr +1, 1, grid.nr +1, grid.nr, grid.nr -1, 0)
    diagonals_cropped = (
        diagonals[1:, 0],  # left
        diagonals[grid.nr+1:, 1],  # lower left
        diagonals[grid.nr:, 2],  # lower
        diagonals[grid.nr-1:, 3],  # lower right
        diagonals[:-1, 4],  # right
        diagonals[:-grid.nr-1, 5],  # upper right
        diagonals[:-grid.nr, 6],  # upper
        diagonals[:-grid.nr+1, 7],  # upper left
        diagonals[:, 8],  # center
    )
    derivative = sparse.diags(diagonals_cropped, offsets, format='csc', shape=(grid.size, grid.size))
    if mask is not None:
        derivative = reduce_matrix(derivative, mask)
    if compensate_edges:
        derivative = compensate_matrix(derivative)
    return derivative
