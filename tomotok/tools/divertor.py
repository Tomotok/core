# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
"""
"""
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter
from scipy.sparse import sparray

from tomotok.core.geometry import RegularGrid


def get_divertor_channels(
    grid: RegularGrid,
    divertor_area_r: ArrayLike,
    divertor_area_z: ArrayLike,
    gmat: np.ndarray | sparray,
    method: str = 'any',
) -> NDArray[np.bool_]:
    """
    Finds channels of geometry matrix that have contributions in the divertor area.
    
    Uses dot product of geometry matrix with divertor area mask matrix obtained from :func:`divertor_area_mask`.

    Parameters
    ----------
    grid : RegularGrid
    divertor_area_r : ArrayLike
        radial coordinates of divertor area boundary polygon
    divertor_area_z : ArrayLike
        vertical coordinates of divertor area boundary polygon
    gmat : numpy.ndarray or scipy.sparse.sparray
        geometry matrix with shape (# channels, # nodes)
    method : str, optional
        Method for defining divertor area mask. Supported values are 'any' and 'centers'. Default value is 'any'.

    Returns
    -------
    numpy.ndarray of bool
        mask array with True on channels interfering with predefined divertor area
    
    See also
    --------
    :meth:`grid.is_inside`, :meth:`grid.is_inside_any`
    """
    if method == 'any':
        divertor_mask = grid.is_inside_any(divertor_area_r, divertor_area_z)
    elif method == 'centers':
        divertor_mask = grid.is_inside(divertor_area_r, divertor_area_z)
    else:
        raise ValueError(f'Unsupported method {method} for defining divertor area mask. Supported values are "any" and "centers".')
    dotp = gmat @ divertor_mask.flatten()
    mask = dotp.sum(axis=1) > 0
    return mask


def divertor_weighted_matrix(
    grid: RegularGrid, 
    divertor_area_r: ArrayLike,
    divertor_area_z: ArrayLike,
    divertor_weight=0.4, 
    border_matrix: tuple[np.ndarray, np.ndarray] | None= None,
    outside_weight: float = 0.1,
    outside_weight_sigma: float = 3,
    method: str = 'any',
) -> np.ndarray:
    """
    Computes weight matrix. Supports different weight for pixels located in divertor area or those that are
    outside vacuum vessel.

    This matrix is useful for regularization of inverse problem, where the emission from divertor area is expected to
    differ from the rest of plasma. 

    Parameters
    ----------
    grid : RegularGrid
    divertor_area_r : ArrayLike
        Radial coordinates of divertor area boundary polygon.
    divertor_area_z : ArrayLike
        Vertical coordinates of divertor area boundary polygon.
    divertor_weight : float, optional
        Specifies divertor pixels' weight. Default value for divertor is 0.4 and standard weight of pixel is 1.
        If standard weight is used, the divertor area will not have any preference.
    border_matrix : tuple of numpy.ndarray, optional
        Coordinates of vacuum vessel border outline in (r, z). If provided, the pixels outside of vacuum vessel will be assigned with outside_weight.
    outside_weight : float, optional
        Specifies weight for pixels outside of vacuum vessel, that is used for smoothing the boundary.
        Inside nodes are defined by border_matrix parameter. Default value is 0.1
    outside_weight_sigma : float, optional
        Sigma for gaussian filter used for smoothing the boundary between inside and outside of vacuum vessel. Default value is 3.
    method : str, optional
        Method for defining divertor area mask. Supported values are 'any' and 'centers'. Default value is 'any'.

    Returns
    -------
    numpy.ndarray dtype float64
        matrix with pixel weights with same shape as grid (#y pixels, #x pixels)
    """
    wm = np.ones(grid.shape)
    if divertor_weight != 1:
        if method == 'any':
            divertor_area = grid.is_inside_any(divertor_area_r, divertor_area_z)
        elif method == 'centers':
            divertor_area = grid.is_inside(divertor_area_r, divertor_area_z)
        else:
            raise ValueError(f'Unsupported method {method} for defining divertor area mask. Supported values are "any" and "centers".')
        wm[divertor_area] = divertor_weight
    if border_matrix is not None:
        wm = gaussian_filter(wm, sigma=outside_weight_sigma, mode='nearest')
        wm = np.where(border_matrix, wm, outside_weight)
    return wm
