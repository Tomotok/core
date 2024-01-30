# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Contains functions that can create emissivity phantoms.

Both isotropic one or based on the shape of flux surfaces.
"""
from typing import Union, Tuple

import numpy as np


def elliptical_flux(radial_num: int, vertical_num: int, span: Union[float, Tuple[float, float]] = 1.5) -> np.ndarray:
    """
    Creates matrix of artificial flux surfaces with elliptical shape.

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


def iso_psi(nx, ny, span=1.5):
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
    x = np.linspace(-span, span, nx)
    y = np.linspace(-span, span, ny)
    mx, my = np.meshgrid(x, y)
    res = np.sqrt(mx * mx + my * my)
    return res


def gauss(x, w=.1, lim=1, amp=1, cen=0.):
    r"""
    Creates anisotropic gaussian artificial emissivity by 1D transform of x

    .. math::
        f = amp \left( \mathrm{e}^{-(x-cen)^2 / w } - \mathrm{e}^{-(lim-cen)^2 / w)} \right)

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
    tx = (x - cen) * (x - cen)
    tlim = (lim - cen) * (lim - cen)
    res = amp * (np.exp(-tx / w) - np.exp(-tlim / w))
    res = np.where(res < 0, 0, res)
    return res


def gauss_iso(nx, ny, span=1.2, w=.1, lim=1, amp=1, cen=0):
    """
    Creates isotropic gaussian distribution.
    See references for iso_psi and gauss
    """
    x = iso_psi(nx, ny, span)
    res = gauss(x, w, lim, amp, cen)
    return res


def polar_phase(x, num=3, shift=0):
    """
    Applies sine phase in radial angle direction to given profile x.

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
    # TODO pixels do not have to be sqaures
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
    # TODO consider using absolute value of sin
    multip = np.sin(num * angle + shift)
    res = x * multip
    return res


def islands(psi, w=.01, lim=1, amp=1, cen=0.4, num=3, shift=0):
    """
    Creates island like phantom from given psi profile. See references for gauss and polar_phase.

    See also
    --------
    gauss, polar_phase
    """
    res = gauss(psi, w, lim, amp, cen)
    res = polar_phase(res, num, shift)
    return res
