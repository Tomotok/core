# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Routines for generating line of sight start and end points.
"""
import json
from pathlib import Path
from typing import Tuple, Union, List

import numpy as np
from numpy.typing import ArrayLike


NumberType = Union[int, Tuple[int, int]]
FovType = Union[float, Tuple[float, float]]
VectorType = Union[Tuple[float, float, float], np.ndarray]


def generate_directions(
    num: NumberType = (10, 1), 
    fov: FovType = (45, 0), 
    axis: VectorType = (1, 0, 0), 
    length=1.0
) -> np.ndarray:
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
    length : float, optional
        line of sight length, by default 1

    Returns
    -------
    dirs : numpy.ndarray
        contains direction vectors cartesian coordinates, shape (#los, 3)
    """
    if isinstance(fov, (int, float)):
        fov = (fov, 0)
    fov = np.deg2rad(fov)
    if isinstance(num, int):
        num = (num, 1)
    ntot = num[0] * num[1]

    dirs = np.full((ntot, 3), length, dtype=float)

    ve = length * np.tan(fov[0] / 2)
    ye = length * np.tan(fov[1] / 2)
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


def generate_los(
    num: NumberType = (10, 1),
    fov: FovType = (45, 0), 
    axis: VectorType = (1, 0, 0), 
    length=1.0,
    pinhole: VectorType = (0, 0, 0), 
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates line of sight endpoints with uniform distribution.

    Parameters
    ----------
    num : int or (int,int), optional
        number of generated chords (vertical, horizontal)
    fov : float, or tuple of two float, optional
        vertical and horizontal field of view in degrees
    axis : tuple of three floats, optional
        direction of chordal axis
    length : float, optional
        line of sight length, by default 1
    pinhole : tuple of three floats, optional
        r and z coordinates of pinhole
        FIXME: pinhole description

    Returns
    -------
    start : numpy.ndarray
        array with line of sight start points coordinates, shape (#los, 3)
    end : numpy.ndarray
        array with line of sight end points coordinates, shape (#los, 3)
    """
    directions = generate_directions(num, fov, axis, length)
    start = np.full_like(directions, pinhole)
    # FIXME: pinhole handling
    end = directions + pinhole
    return start, end


def rot_v(points: ArrayLike, angle: float) -> np.ndarray:
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
    -------
    numpy.ndarray
        array of rotated points
    """
    s = np.sin(angle)
    c = np.cos(angle)
    mat = np.array(((c, 0, s),
                    (0, 1, 0),
                    (-s, 0, c)))
    rpoints = points @ mat
    return rpoints


def rot_h(points: ArrayLike, angle: float) -> np.ndarray:
    """
    Rotates given points in horizontal direction, that is about vertical axis z.

    Parameters
    ----------
    points : numpy.ndarray 
        3D coordinates of points to be rotated with shape (#points, 3)
    angle : float
        angle of rotation in radians

    Returns
    -------
    numpy.ndarray
        array of rotated points
    """
    s = np.sin(angle)
    c = np.cos(angle)
    mat = np.array(((c, s, 0),
                    (-s, c, 0),
                    (0, 0, 1)))
    rpoints = points @ mat
    return rpoints


def save_los(
        loc: Union[str, Path], 
        startpoints: Union[ArrayLike, List[ArrayLike]], 
        endpoints: Union[ArrayLike, List[ArrayLike]], 
        detector_names: Union[str, List[str]] = None
):
    """
    Saves line of sight start and end points to json file.

    Parameters
    ----------
    loc : str or Path
        location of los file for saving
    startpoints, endpoints : list of numpy.ndarray
        list of arrays with start and end points coordinates, shape (#chords, 3)
    detector_names : list of str, optional
        list of detector names, `detector_#` is used if not specified
    """
    loc = Path(loc).expanduser()
    startpoints = np.array(startpoints)
    endpoints = np.array(endpoints)
    if endpoints.shape != startpoints.shape:
        raise ValueError('Start and end points lists must have same length and shape.')
    if startpoints.ndim == 3:
        points_number = len(startpoints)
    elif startpoints.ndim == 2:
        points_number = 1
        startpoints = startpoints[None, ...]
        endpoints = endpoints[None, ...]
    else:
        raise ValueError('Start and end points arrays must be 2D arrays.')

    if detector_names is None:
        detector_names = [f'detector_{i}' for i in range(points_number)]
    elif isinstance(detector_names, str):
        detector_names = [detector_names]
    if len(detector_names) != points_number:
        raise ValueError('Number of detector names must match number of start/end points.')

    los = {}
    for i, name in enumerate(detector_names):
        los[name] = {
            'startpoints': startpoints[i].tolist(), 
            'endpoints': endpoints[i].tolist()
        }

    with open(loc, 'w') as fl:
        json.dump(los, fl)

    return
