# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Contains utility functions for creation of emissivity phantoms.

The phantoms are based on a gaussian profile as a function of magnetic flux coordinates.
A function for creation of a simple flux map with elliptical shape is also provided.
Additionally, a function for creation of a polar phase matrix is provided, 
which can be used to create magnetic island-like structures in the emissivity phantom.
"""
from warnings import warn

import numpy as np

from tomotok.geometry import RegularGrid


def regular_elliptical_flux(
    grid: RegularGrid, span: float | tuple[float, float] = 1.5
) -> np.ndarray:
    """Creates a matrix of artificial flux surfaces with elliptical shape based on provided grid."""
    return elliptical_flux(grid.nr, grid.nz, span)


def elliptical_flux(
    radial_num: int, vertical_num: int, span: float | tuple[float, float] = 1.5
) -> np.ndarray:
    """
    Creates a matrix of artificial flux surfaces with elliptical shape.

    Minimum values at the border are determined by the span parameter.

    Parameters
    ----------
    radial_num : int
        number of nodes along radial axis
    vertical_num : int
        number of nodes along vertical axis
    span : float or tuple of two floats, optional
        flux value at the center of grid edge
        if tuple, first value is for radial axis, second for vertical
    
    Returns
    -------
    numpy.ndarray
        Matrix with generated fluxes
    """
    if isinstance(span, tuple):
        if len(span) != 2:
            raise ValueError('Span parameter must be float or tuple of two floats.')
        span_r, span_v = span
    else:
        span_r = span_v = span
    radial = np.linspace(-span_r, span_r, radial_num)
    vertical = np.linspace(-span_v, span_v, vertical_num)
    radial, vertical = np.meshgrid(radial, vertical)
    fluxes = np.sqrt(radial ** 2 + vertical ** 2)
    return fluxes


def gaussian_on_flux(
    flux: np.ndarray, amplitude: float = 1, center: float = 0, width: float = 0.1, 
    limit: float = 1, limit_width: float = 0.2, limit_power: int = 2
) -> np.ndarray:
    r"""
    Creates gaussian artificial emissivity profile by transforming provided flux values

    .. math::
        f = a \mathrm{e}^{-(f - c)^2 / w }

    The limit width parameter allows to create a smooth transition from gaussian profile to zero at the limit value of flux.
    This is done by multiplying the gaussian profile with a polynomial function that goes from 1 to 0 in the range of limit - limit_width to limit.

    Parameters
    ----------
    flux : np.ndarray
        Flux values for transformation, any shape is supported
    amplitude : float, optional
        maximum of gaussian profile, 
    center : float, optional
        center of gaussian profile
        allows hollow profile generation when mapped on psi
    width : float, optional
        width of gaussian profile    
    limit : float, optional
        flux value where emissivity is forced to reach zero
        if flux > limit emissivity is set to zero
    limit_width : float, optional
        width of transition from gaussian profile to zero at limit value of flux
    limit_power : float, optional
        power of polynomial transition from gaussian profile to zero at limit value of flux

    Returns
    -------
    numpy.ndarray
        Transformed values of x with same dimensions
    """
    center_dst_sq = (flux - center) * (flux - center)
    res = np.exp(-center_dst_sq / width)

    lim_start = limit - limit_width
    edge_poly = ((flux - lim_start)/ limit_width) ** limit_power
    limit_value = np.exp(-(limit - center)**2 / width)
    result_modifier = np.where(flux < lim_start, 0, limit_value * edge_poly)
    res -= result_modifier
    res = np.where(res < 0, 0, res)
    res *= amplitude
    return res


def polar_phase_matrix(
    grid: RegularGrid, num: int = 3, shift: float = 0.0,
    center: tuple[float, float] | None = None
) -> np.ndarray:
    """
    Creates a matrix with sine phase in poloidal direction relative to provided center.

    The intended use is to modify an emissivity pattern by multiplying it with the returned matrix.
    This could be used to create island-like radiating structures in the phantom.
    The phase is applied in the positive angle direction (counterclockwise).
    If center is not provided, the grid centre is used.

    Parameters
    ----------
    grid : RegularGrid
        grid for which the phase matrix is generated
    num : int, optional
        number of periods per one rotation
    shift : float, optional
        initial phase shift in degrees
    center : tuple of two floats, optional
        center of polar phase, if None, grid centre is used
    """
    center = center or grid.centre
    shift = np.deg2rad(shift)
    mesh_r, mesh_z = grid.center_mesh
    diff_r = mesh_r - center[0]
    diff_z = mesh_z - center[1]
    angle = np.arctan2(diff_z, diff_r)
    out = np.sin(num * (angle + shift))
    return out


def iso_psi(nx: int, ny: int, span: float = 1.5) -> np.ndarray:
    """
    Creates matrix of artificial isotropic psi profile with border values for
    each axis equal to span.

    .. deprecated:: 2.0
        Use :func:`elliptical_flux` instead.

    Parameters
    ----------
    nx : int
        Number of pixels on x axis
    ny : int
        Number of pixels on y axis
    span : float, optional
        Value of result on the center of border

    Returns
    -------
    numpy.ndarray
        Matrix with generated profile
    """
    warn(
        "Function iso_psi is deprecated and will be removed in future versions. " +
        "Use `elliptical_flux` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    x = np.linspace(-span, span, nx)
    y = np.linspace(-span, span, ny)
    mx, my = np.meshgrid(x, y)
    res = np.sqrt(mx * mx + my * my)
    return res


def gauss(
    x: np.ndarray, w: float = 0.1, lim: float = 1, amp: float = 1, cen: float = 0.0,
) -> np.ndarray:
    r"""
    Creates anisotropic gaussian artificial emissivity by 1D transform of x

    .. math::
        f = amp \left( \mathrm{e}^{-(x-cen)^2 / w } - \mathrm{e}^{-(lim-cen)^2 / w)} \right)
        
    .. deprecated:: 2.0
        Use :func:`gaussian_on_flux` instead.

    Can be used on np.ndarray. Lim should be greater than cen.

    Parameters
    ----------
    x : float, array, np.ndarray
        Contains values to be transformed, usually psi
    w : float, optional
        width of gaussian profile    
    lim : float, optional
        minimal value of x where transform gives zero
        if x > lim emissivity is set to zero    
    amp : float, optional
        amplitude of gaussian profile, 
    cen : float, optional
        center of gaussian profile
        allows hollow profile generation when mapped on psi

    Returns
    -------
    numpy.ndarray
        Transformed values of x with same dimensions
    """
    warn(
        "Function `gauss` is deprecated and will be removed in future versions. " +
        "Use `gaussian_on_flux` instead.", 
        DeprecationWarning,
        stacklevel=2
    )
    tx = (x - cen) * (x - cen)
    tlim = (lim - cen) * (lim - cen)
    res = amp * (np.exp(-tx / w) - np.exp(-tlim / w))
    res = np.where(res < 0, 0, res)
    return res


def gauss_iso(
    nx: int, ny: int, 
    span: float = 1.2, w: float = 0.1, lim: float = 1, amp: float = 1, cen: float = 0.0,
) -> np.ndarray:
    """
    Creates isotropic gaussian distribution.
    See references for iso_psi and gauss

    .. deprecated:: 2.0
        Use :func:`gaussian_on_flux` instead.
    """
    warn(
        "Function gauss_iso is deprecated and will be removed in future versions. " +
        "Use `gaussian_on_flux` instead.", DeprecationWarning,
        stacklevel=2
    )
    x = iso_psi(nx, ny, span)
    res = gauss(x, w, lim, amp, cen)
    return res


def polar_phase(x: np.ndarray, num: int = 3, shift: float = 0) -> np.ndarray:
    """
    Applies sine phase in radial angle direction to given profile x.

    Assumes equal dimension of grid elements in both axes. 

    .. deprecated:: 2.0
        Use :func:`polar_phase_matrix` instead.

    Parameters
    ----------
    x : numpy.ndarray
        Profile for application of polar phase.
    num : int, optional
        number of periods per one rotation
    shift : float, optional
        initial phase shift

    Returns
    -------
    numpy.ndarray
        Matrix with applied polar phase
    """
    warn(
        "Function polar_phase is deprecated and will be removed in future versions. " + 
        "Use `polar_phase_matrix` instead.", 
        DeprecationWarning,
        stacklevel=2,
    )
    rstep = 1
    cstep = 1
    im = np.argwhere(x == np.min(x))
    im = np.around(np.average(im, 0))
    r, c = np.shape(x)
    mr, mc = np.meshgrid(np.arange(c), np.arange(r))
    dr = (mr - im[1]) * rstep
    dc = (mc - im[0]) * cstep
    dst = np.sqrt(dr ** 2 + dc ** 2)
    dst[int(im[0]), int(im[1])] = -1
    angle = np.arcsin(dc / dst)
    angle = np.where(np.sign(dr) > 0, np.pi - angle, angle)
    multip = np.sin(num * angle + shift)
    res = x * multip
    return res


def islands(
    psi: np.ndarray, 
    w: float = 0.01, lim: float = 1, amp: float = 1, cen: float = 0.4, num: int = 3, 
    shift: float = 0.0,
) -> np.ndarray:
    """
    Creates island like phantom from given psi profile. See references for gauss and polar_phase.

    .. deprecated:: 2.0
        Use :func:`gaussian_on_flux` and :func:`polar_phase_matrix` instead

    See also
    --------
    gauss, polar_phase
    """
    warn(
        "Function islands is deprecated and will be removed in future versions. " +
        "Use `gaussian_on_flux` and `polar_phase_matrix` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    res = gauss(psi, w, lim, amp, cen)
    res = polar_phase(res, num, shift)
    return res
