# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Handles computation of derivative matrices used for regularization in MFR algorithm.
"""
import numpy as np
from scipy import sparse

from tomotok.core.geometry import RegularGrid


def all_direction_derivative_matrices(
    grid: RegularGrid,
    scheme: str = 'forward',
    mask: np.ndarray | None = None,
    compensate_edges: bool = True,
    compensation_fraction: float = 1,
) -> list[sparse.csc_array]:
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
    compensation_fraction : float, optional
        Fraction of diagonal elements to be subtracted, by default 1.

    Returns
    -------
    list of scipy.sparse.csc_matrix
        List of derivative matrices for all 8 directions with following order:
        right, top-right, top, top-left, left, bottom-left, bottom, bottom-right
    """
    kw = dict(scheme=scheme, mask=mask, compensate_edges=compensate_edges, compensation_fraction=compensation_fraction)
    derivatives = [
        derivative_matrix(grid, 'right', **kw),
        derivative_matrix(grid, 'top-right', **kw),
        derivative_matrix(grid, 'top', **kw),
        derivative_matrix(grid, 'top-left', **kw),
        derivative_matrix(grid, 'left', **kw),
        derivative_matrix(grid, 'bottom-left', **kw),
        derivative_matrix(grid, 'bottom', **kw),
        derivative_matrix(grid, 'bottom-right', **kw),
    ]
    return derivatives


def standard_anisotropic_derivative_matrices(
    grid: RegularGrid,
    flux: np.ndarray,
    mask: np.ndarray | None = None,
    compensate_edges: bool = True,
    compensation_fraction: float = 1,
) -> list[sparse.csc_array]:
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
    compensation_fraction : float, optional
        Fraction of row sum to subtract from diagonal, by default 1.

    Returns
    -------
    list of scipy.sparse.csc_matrix
        List of derivative matrices for parallel and perpendicular directions in following order:
        parallel counter clockwise, perpendicular counter clockwise, parallel clockwise, perpendicular clockwise
    """
    kw = dict(mask=mask, compensate_edges=compensate_edges, compensation_fraction=compensation_fraction)
    derivatives = [
        anisotropic_derivative_matrix(grid, flux, 'parallel', 'forward', **kw),
        anisotropic_derivative_matrix(grid, flux, 'perpendicular', 'forward', **kw),
        anisotropic_derivative_matrix(grid, flux, 'parallel', 'backward', **kw),
        anisotropic_derivative_matrix(grid, flux, 'perpendicular', 'backward', **kw),
    ]
    return derivatives


def reduce_matrix(mat: sparse.sparray, mask: np.ndarray) -> sparse.sparray:
    """
    Creates reduced matrix by cutting out rows and columns representing unwanted nodes.

    Parameters
    ----------
    mat : scipy.sparse.sparray
        matrix to be reduced
    mask : numpy.ndarray of bool
        mask array for selecting desired nodes, True for nodes to keep
        must be 1D or 2D, size must match grid size (# rows or # columns)

    Returns
    -------
    scipy.sparse.sparray
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


def compensate_matrix(mat: sparse.sparray, compensation_fraction: float = 1) -> sparse.sparray:
    """
    Subtracts from diagonal so that sum of each row is zero.

    Parameters
    ----------
    mat : scipy.sparse.sparray
        matrix to be compensated
    compensation_fraction : float, optional
        Fraction of row sum to subtract from diagonal, by default 1.
        1 corresponds to full compensation where row sum should be zero.
        However, the compensation may cause small negative values due to rounding errors.
        Using slightly smaller value (e.g. 0.9999) can prevent this issue while still effectively acting as a zero row sum constraint.

    Returns
    -------
    scipy.sparse.sparray
        compensated matrix
    """
    row_sum = np.array(mat.sum(1)).flatten()
    row_sum_diag = sparse.diags_array(row_sum, format=mat.format)
    mat = mat - compensation_fraction * row_sum_diag
    return mat


def derivative_matrix(
    grid: RegularGrid,
    direction: str,
    scheme: str = 'forward',
    mask: np.ndarray | None = None,
    compensate_edges: bool = True,
    compensation_fraction: float = 1,
) -> sparse.csc_array:
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
    compensation_fraction : float, optional
        Fraction of row sum to subtract from diagonal, by default 1.

    Returns
    -------
    scipy.sparse.csc_array

    See Also
    --------
    all_direction_derivative_matrices : creates derivative matrices for all 8 directions
    reduce_matrix : reduces matrix by cutting out rows and columns representing unwanted nodes
    compensate_matrix : subtracts from diagonal so that sum of each row is zero or nearly zero
    """
    kw_diags = dict(format='csc', shape=(grid.size, grid.size))
    value_diags = [1.0]
    center = sparse.diags_array(value_diags, offsets=[0], **kw_diags)
    right = sparse.diags_array(value_diags, offsets=[1], **kw_diags)
    upper_right = sparse.diags_array(value_diags, offsets=[grid.nr + 1], **kw_diags)
    upper = sparse.diags_array(value_diags, offsets=[grid.nr], **kw_diags)
    upper_left = sparse.diags_array(value_diags, offsets=[grid.nr - 1], **kw_diags)
    left = sparse.diags_array(value_diags, offsets=[-1], **kw_diags)
    lower_left = sparse.diags_array(value_diags, offsets=[-grid.nr - 1], **kw_diags)
    lower = sparse.diags_array(value_diags, offsets=[-grid.nr], **kw_diags)
    lower_right = sparse.diags_array(value_diags, offsets=[-grid.nr + 1], **kw_diags)

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
        msg = 'Unknown numerical derivative scheme {}. Use one of {}.'
        raise ValueError(msg.format(scheme, schemes))

    # normalisation
    dmat = dmat / step
    if mask is not None:
        dmat = reduce_matrix(dmat, mask)
    if compensate_edges:
        dmat = compensate_matrix(dmat, compensation_fraction=compensation_fraction)
    return dmat


def laplace_matrix(
    grid: RegularGrid,
    mask: np.ndarray | None = None,
    compensate_edges: bool = True,
    compensation_fraction: float = 1.0,
    diagonals: bool = True,
) -> sparse.csc_array:
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
    kw_diags = dict(format='csc', shape=(grid.size, grid.size))
    value_diags = [1.0]
    center = sparse.diags_array(value_diags, offsets=[0], **kw_diags)
    right = sparse.diags_array(value_diags, offsets=[1], **kw_diags)
    upper = sparse.diags_array(value_diags, offsets=[grid.nr], **kw_diags)
    left = sparse.diags_array(value_diags, offsets=[-1], **kw_diags)
    lower = sparse.diags_array(value_diags, offsets=[-grid.nr], **kw_diags)

    lmat = (right + left) / grid.dr
    lmat += (upper + lower) / grid.dz
    lmat -= (2 / grid.dr + 2 / grid.dz) * center

    if diagonals:
        upper_right = sparse.diags_array(value_diags, offsets=[grid.nr + 1], **kw_diags)
        upper_left = sparse.diags_array(value_diags, offsets=[grid.nr - 1], **kw_diags)
        lower_left = sparse.diags_array(value_diags, offsets=[-grid.nr - 1], **kw_diags)
        lower_right = sparse.diags_array(value_diags, offsets=[-grid.nr + 1], **kw_diags)
        diagonal_distance = (grid.dr**2 + grid.dz**2)**0.5

        lmat += (upper_right + upper_left + lower_left + lower_right) / diagonal_distance
        lmat -= 4 / diagonal_distance * center

    if mask is not None:
        lmat = reduce_matrix(lmat, mask)
    if compensate_edges:
        lmat = compensate_matrix(lmat, compensation_fraction=compensation_fraction)
    return lmat


def anisotropic_derivative_matrix(
    grid: RegularGrid,
    flux: np.ndarray,
    direction: str = 'parallel',
    scheme: str = 'forward',
    mask: np.ndarray | None = None,
    compensate_edges: bool = True,
    compensation_fraction: float = 1.0
) -> sparse.csc_array:
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
    flux : numpy.ndarray
        Matrix with magnetic flux values, shape has to match grid
    mask : numpy.ndarray of bool, optional
        A bool mask determining nodes of regular grid to keep, by default None.
    direction : str, optional
        Direction of derivative computation, by default 'parallel'
    scheme : str, optional
        derivative scheme used, by default 'forward'
         - forward is in the direction of positive angle i.e. counter clockwise
         - backward is in the direction of negative angle i.e. clockwise

    Returns
    -------
    scipy.sparse.csc_matrix

    See Also
    --------
    standard_anisotropic_derivative_matrices : creates derivative matrices for parallel and perpendicular directions
    reduce_matrix : reduces matrix by cutting out rows and columns representing unwanted nodes
    compensate_matrix : subtracts from diagonal so that sum of each row is zero or nearly zero
    """
    if grid.shape != flux.shape:
        raise ValueError('Grid shape does not match fluxes shape.')
    if scheme not in ['forward', 'backward']:
        raise ValueError('Scheme must be either `forward` or `backward`.')
    if direction not in ['parallel', 'perpendicular']:
        raise ValueError('Direction must be either `parallel` or `perpendicular`.')

    grad_z, grad_r = np.gradient(flux, grid.z_center, grid.r_center)
    atan2 = np.arctan2(grad_z, grad_r)  # direction angle of flux surface gradient in radians
    atan_mod = (atan2.flatten() - atan2.min()) / (2 * np.pi) * 8  # rescale atan2 to <0, 8) range

    rotation = 0
    if direction == 'parallel':
        rotation += 2  # rotate gradient by 90 degrees (2 * pi/4) to get tangent
    if scheme == 'backward':
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
    derivative = sparse.diags_array(
        diagonals_cropped, 
        offsets=offsets,
        format='csc', 
        shape=(grid.size, grid.size),
    )
    if mask is not None:
        derivative = reduce_matrix(derivative, mask)
    if compensate_edges:
        derivative = compensate_matrix(derivative, compensation_fraction=compensation_fraction)
    return derivative
