# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Routines for generation of line of sight start and end points.

Creates the lines of sights in x axis direction and then rotates them about the other horizontal axis, then about vertical axis.
"""
import numpy as np


def generate_directions(num=(10, 1), fov=(45, 0), axis=(1, 0, 0), elong=1.):
    """
    Creates direction vectors for lines of sight using camera like convention.

    The first line of sight is top left, then the numbering follows a row, the last one is bottom right.

    Parameters
    ----------
    num : int or (int,int), optional
        number of generated chords (vertical, horizontal)
    fov : float, or tuple of two float, optional
        vertical or (vertical, horizontal) field of view in degrees
    axis : tuple of three floats, optional
        direction of symmetry axis
    elong : float, optional
        vector length multiplier

    Returns
    -------
    dirs : numpy.ndarray
        contains direction vectors cartesian coordinates, shape (#los, 3)
    """
    if isinstance(fov, (int, float)):
        fov = (fov, 0)
    fov = np.deg2rad(fov)
    try:
        ntot = num[0] * num[1]
    except TypeError:
        ntot = num
        num = (num, 1)

    dirs = np.full((ntot, 3), elong, dtype=np.double)

    ve = elong * np.tan(fov[0] / 2)
    ye = elong * np.tan(fov[1] / 2)
    vg = np.linspace(ve, -ve, num[0])
    yg = np.linspace(-ye, ye, num[1])

    ym, vm = np.meshgrid(yg, vg)
    dirs[:, 1] = ym.flatten()
    dirs[:, 2] = vm.flatten()

    nrm = np.linalg.norm(axis)
    try:
        axis = axis / nrm
    except ZeroDivisionError:
        raise ValueError('Axis has to be a non zero vector.')
    vo = np.arcsin(axis[2])  # vertical offset angle
    if axis[1] == axis[0] == 0:  # vertical vector
        dh = 0
    elif axis[0] == 0:  # horizontal in y-axis direction
        dh = axis[1] * np.inf
    else:  # general direction
        dh = np.sign(axis[1]) * np.abs(axis[1] / axis[0])
    ho = np.arctan(dh)
    dirs = rot_v(dirs, vo)
    dirs = rot_h(dirs, ho)
    if axis[0] < 0:
        dirs[:, 0] = -dirs[:, 0]
    return dirs


def generate_los(pinhole=(0, 0, 0), num=(10, 1), fov=(45, 0), axis=(1, 0, 0), elong=1.):
    """
    Creates line of sight endpoints with uniform distribution.

    Parameters
    ----------
    num : int or (int,int), optional
        number of generated chords (vertical, horizontal)
    pinhole : tuple of three floats, optional
        r and z coordinates of pinhole
    fov : float, or tuple of two float, optional
        vertical and horizontal field of view in degrees
    axis : tuple of three floats, optional
        direction of chordal axis
    elong : float, optional
        chord length multiplier

    Returns
    -------
    start : numpy.ndarray
        array with line of sight start points coordinates, shape (#los, 3)
    end : numpy.ndarray
        array with line of sight end points coordinates, shape (#los, 3)
    """
    directions = generate_directions(num, fov, axis, elong)
    start = np.full_like(directions, pinhole)
    end = directions + pinhole
    return np.array((start, end))


def rot_v(points, angle):
    """
    Rotates given points in vertical direction, that is about horizontal y axis perpendicular to r/x.

    Should be done before horizontal rotation.

    Parameters
    ----------
    points : numpy.ndarray 
        3D coordinates of points to be rotated with shape (#points, 3)
    angle : float
        angle of rotation in radians

    Returns
    --------
    numpy.ndarray
        array of rotated points
    """
    s = np.sin(angle)
    c = np.cos(angle)
    mat = np.array(((c, 0, s),
                    (0, 1, 0),
                    (-s, 0, c)))
    rpoints = points.dot(mat)
    return rpoints


def rot_h(points, angle):
    """
    Rotates given points in horizontal direction, that is about vertical axis z.

    Parameters
    -----------
    points : numpy.ndarray 
        3D coordinates of points to be rotated with shape (#points, 3)
    angle : float
        angle of rotation in radians

    Returns
    --------
    numpy.ndarray
        array of rotated points
    """
    s = np.sin(angle)
    c = np.cos(angle)
    mat = np.array(((c, s, 0),
                    (-s, c, 0),
                    (0, 0, 1)))
    rpoints = points.dot(mat)
    return rpoints
