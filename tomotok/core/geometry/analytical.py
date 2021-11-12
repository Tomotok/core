# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Contains analytical geometry matrix generators. Requires shapely library.

Shapely module is imported after the function is called so that it is not a hard dependence.
These are significantly slower than numerical ones, but can use polygons instead of line of sights.
"""
import time
import warnings

import numpy as np


def shapely_line_mat(xchord, ychord, grid):
    """
    Generates geometry matrix using shapely module
    
    Parameters
    ----------
    xchord, ychord : numpy.ndarray
        chord coordinates with shape (#chords, 2)
    grid : RegularGrid
        class of tomotok module describing reconstruction area

    Returns
    -------
    gmat : dict
        holds geometry matrix and grid coordinates
    """
    warnings.warn('Shapely based gmat generators will be removed in the future.', FutureWarning)
    import shapely.geometry as sg
    nx, ny = grid.nr, grid.nz
    xg, yg = grid.r_border, grid.z_border
    st = time.time()
    nch = xchord.shape[0]
    # cdct = dict(chnl=np.arange(nch), y=grid.z_center, x=grid.r_center)
    gmat = dict()
    gmat['values'] = np.zeros((nch, ny, nx))
    gmat['z'] = grid.z_center
    gmat['r'] = grid.r_center
    gmat['type'] = 'line'
    lines = []
    complex_chord = xchord.shape[1] > 2
    for i in range(nch):
        if complex_chord:
            crop = (xchord[i] < xg[-1]) * (xchord[i] > xg[0])
        else:
            crop = np.ones(xchord.shape[1], dtype=np.bool)
        coords = np.stack((xchord[i, crop], ychord[i, crop]), axis=1)
        line = sg.LineString(coords)
        lines.append(line)
    for row in range(ny):
        for col in range(nx):
            pix = sg.box(xg[col], yg[row], xg[col+1], yg[row+1])
            for i, line in enumerate(lines):
                gmat['values'][i, row, col] = pix.intersection(line).length
        if row % 10 == 9:
            ela = time.time() - st
            print('finished row {}/{}'.format(row + 1, ny),
                  'elapsed time {:.2f}s'.format(ela))
    ela = time.time() - st
    print('Gmat generation time {:.2f}s'.format(ela),
          'average time per chord {:.0f}ms'.format(ela/nch*1000))
    return gmat

# TODO remove
def shapely_poly_mat(xchord, ychord, grid, ph=None, w=None):
    """
    Generates geometry matrix using polygon object from shapely module.
    
    Parameters
    ----------
    xchord, ychord : numpy.ndarray
        chord coordinates with shape (#chords, #2) or (#chords, #curve_points)
    grid : RegularGrid
        class of tomotok module describing reconstruction area
    ph : numpy.ndarray, optional
        matrix with pinholes coords with shape (#chords, 2)
        If no values are supplied, start points of chords are used.
    w : numpy.ndarray, optional
        contains widening coefficients for chords with shape (#chords,).
        If no values are supplied, locally predefined value is used.

    Returns
    -------
    dict
        geometry matrix and parameters
    """
    warnings.warn('Shapely based gmat generators will be removed in the future.', FutureWarning)
    import shapely.geometry as sg
    # TODO different widths for each detector
    print('Computing shapely polygon geometry matrix')
    xg = grid.r_border
    yg = grid.z_border
    nx = grid.nr
    ny = grid.nz
    st = time.time()
    nch = xchord.shape[0]
    complex_chord = xchord.shape[1] > 2
    if ph is None or not complex_chord:
        ph = np.stack([xchord[:, 0], ychord[:, 0]], axis=1)
    if w is None:
        const = 0.022 * np.ones(nch)
    elif nch != len(w):
        warnings.warn('Size of provided widening coefficients'
                      ' is different from number of channels.',
                      RuntimeWarning)
        const = np.ones(nch)
        const[:len(w)] = w[:len(const)]
    else:
        const = w
    gmat = dict()
    gmat['values'] = np.zeros((nch, ny, nx))
    gmat['z'] = grid.z_center
    gmat['r'] = grid.r_center
    gmat['type'] = 'poly'
    center = np.array([xg[-1] + xg[0], yg[-1] + yg[0]])/2
    lines = []
    strings = []
    for i in range(nch):
        # xcrop = (xchord[i] < np.max(xg)) * (xchord[i] > np.min(xg))
        # ycrop = (ychord[i] < np.max(yg)) * (ychord[i] > np.min(yg))
        # crop = xcrop * ycrop
        if complex_chord:
            crop = (xchord[i] < xg[-1]) * (xchord[i] > xg[0])
        else:
            crop = np.ones(xchord.shape[1], dtype=np.bool)
        chord = np.stack((xchord[i, crop], ychord[i, crop]), axis=1)
        string = sg.LineString(chord)
        strings.append(string)
        u = chord - ph[i]
        mask = np.dot(u, center - ph[i]) > 0
        u = u[mask]
        n = const[i] * np.fliplr(u) * np.array([-1, 1])
        c1 = chord[mask] + n
        c2 = np.flipud(chord[mask] - n)
        ph_dst = np.linalg.norm(c1 - ph[i], axis=1)
        if ph_dst[0] < ph_dst[-1]:
            coords = np.vstack([ph[i], c1, c2])
        else:
            coords = np.vstack([c1, ph[i], c2])
        poly = sg.Polygon(coords)
        lines.append(poly)
        if not poly.is_valid:
            warnings.warn('invalid polygon for line {}'.format(i),
                          RuntimeWarning)
        # try:
        #     assert poly.is_valid
        #     lines.append(poly)
        # except AssertionError:
        #     wm = ('invalid polygon for line {}'.format(i) +
        #           'Replacing polygon with linestring')
        #     string = sg.LineString(chord)
        #     lines.append(string)
        #     warnings.warn(wm, RuntimeWarning)
    # TODO gamt to gmat['values']
    for row in range(ny):
        for col in range(nx):
            pix = sg.box(xg[col], yg[row], xg[col+1], yg[row+1])
            for i, line in enumerate(lines):
                # isec = pix.intersection(line)
                # area = isec.area
                gmat['values'][i, row, col] = pix.intersection(line).area
        if row % 10 == 9:
            ela = time.time() - st
            print('finished row {}/{}'.format(row + 1, ny),
                  'elapsed time {:.2f}s'.format(ela))
    lengths = lines_len(strings, grid)
    for i in range(nch):
        gmat['values'][i] = gmat['values'][i] / distance_mat(grid, ph[i])
        norm = lengths[i] / gmat['values'][i].sum()
        gmat['values'][i] *= norm
    ela = time.time() - st
    print('Gmat generation time {:.2f}s'.format(ela),
          'average time per chord {:.0f}ms'.format(ela/nch*1000))
    return gmat


def distance_mat(grid, ph):
    """
    Computes distances of pixels centers from pinhole and returns them in a
    matrix form. Indexing starts with bottom row.
    
    Parameters
    ----------
    grid : RegularGrid
        class describing reconstruction area
    ph : numpy.ndarray
        contains pinhole coordinates
    
    Returns
    -------
    numpy.ndarray
        matrix with pixel center distances from pinhole
    """
    x, y = np.meshgrid(grid.r_center - ph[0], grid.z_center - ph[1])
    dst = np.sqrt(np.power(x, 2) + np.power(y, 2))
    return dst

def lines_len(lines, grid):
    """
    Computes lengths of chords inside whole pixel grid

    Parameters
    ----------
    lines : array of shapely.geometry.LineString
        Contains central lines from polygon chord
    grid : RegularGrid

    Returns
    -------
    numpy.ndarray
        array with lengths of chords in pixel grid
    """
    warnings.warn('Shapely based gmat generators and auxiliary functions will be removed in the future.', FutureWarning)
    import shapely.geometry as sg
    box = sg.box(grid.rmin, grid.zmin, grid.rmax, grid.zmax)
    lengths = np.zeros(len(lines))
    for i, line in enumerate(lines):
        lengths[i] = box.intersection(line).length
    return lengths