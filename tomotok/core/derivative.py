# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Contains functions for derivative matrix computation.

 - new functions (isotropic) derivative_matrix and laplace_matrix
 - compute_aniso_dmats and compute_iso_dmats takes care of obtaining
   flux dependent or flux independent derivative matrices,
 - prepare_mag_data is an auxiliary function to calculate magnetic flux gradients and
 - px_norm is a simple auxiliary function to calculate norms for coefficients
"""
import numpy as np
from scipy import sparse
from scipy.sparse import spdiags, eye

from .geometry import RegularGrid


def compute_iso_dmats(grid, derivatives=[1, 2], mask=None):
    """
    Computes isotropic derivative matrix pairs for given differential schemes.

    Parameters
    ----------
    grid : RegularGrid
    mask : np.ndarray, optional
    derivatives : list of int/str, optional

    Returns
    -------
    list of dmat pairs
    """
    if mask is None:
        mask = np.ones(grid.size, dtype=np.bool)
    if isinstance(derivatives, int):
        derivatives = [derivatives]
    dmats = []
    for d in derivatives:
        dmats.append(generate_iso_dmat(grid, derivative=d))
    dmats_red = reduce_dmats(dmats, mask.flatten())
    # dmats_dict = dict()
    # for t in tvec:
    #     dmats_dict[t] = dmats_red
    return dmats_red

def compute_aniso_dmats(grid, magflux, derivative=4, mask=None):
    # TODO change smoothing type to string
    """
    Prepares matrix of finite differences, magnetic configuration dependent or
    fixed simple differences. First or second derivative can be used

    Parameters
    ----------
    grid : RegularGrid
        A reconstruction grid
    magflux : numpy.ndarray
        values of psi normalized mapped to grid
    mask : numpy.ndarray, optional
        bool mask
    derivative : int in {1,2,3,4}
        specifies type of derivative scheme, default is 4

    Returns
    -------
    tuple of scipy.sparse.csrmatrix pairs
        derivative matrix
    """
    # tidx = np.searchsorted(self.magfield['time'], self.tvec)
    # for t in self.tvec:
    #     tidx = np.searchsorted(self.magfield['time'], t)
    #     mf = self.magfield['values'][tidx, :]
    # dmats = compute_aniso_dmats(self.grid, smoothing=1, magflux=mf)

    if mask is None:
        mask = np.ones(grid.size, dtype=np.bool)
    dr = grid.dr
    dz = grid.dz
    npix = grid.nodes_num
    bmat = None
    atan2 = prepare_mag_data(magflux)
    bpar1, bper1, foo, bar = generate_anizo_matrix(grid, atan2, 1)
    bpar2, bper2, foo, bar = generate_anizo_matrix(grid, atan2, 2)

    [br, bz] = generate_iso_dmat(grid, derivative)
    bmat = [bpar1, bper1, bpar2, bper2]
    # correction of boundary conditions => zero derivative on borders !!
    # + ensure positive definiteness
    for i in range(4):
        bmat[i] = bmat[i] - spdiags(bmat[i].sum(axis=1).T, 0, npix, npix)
        ind = np.array(abs(bmat[i]).sum(axis=1).T == 0)
        corr = spdiags(ind, 0, npix, npix)*(br*dr+bz*dz) * spdiags(ind, 0, npix, npix)
        bmat[i] = bmat[i] + corr  # correction adds standard derivatives, probably not necessary
    bmat = [(bmat[0], bmat[1]), (bmat[2], bmat[3])]
    dmats_red = reduce_dmats(bmat, mask.flatten())
    return dmats_red


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
    dn = np.sqrt(2)
    mod = np.asarray([s % 2 for s in direc])
    norms = np.zeros(np.shape(direc))
    norms[mod == 0] = 1
    norms[mod == 1] = dn
    return norms


def generate_anizo_matrix(grid, atan2, derivative):
    # TODO: finish docstrings
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
    ny = grid.nr
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
        dir_ = np.array((-1, 2, 3, 4, 1, -2, -3, -4), dtype=np.int)  # F like
        # dir_ = np.array((-3, -2, 1, 4, 3, 2, -1, -4), dtype=np.int)  # C like
        # C  [-4, -3, -2]       F  [-4, -1,  2]
        #    [-1,  0,  1]          [-3,  0,  3]
        #    [ 2,  3,  4]          [-2,  1,  4]
        # dir_ = np.array((-1, ny-1, ny, ny+1, 1, -ny+1, -ny, -ny-1), dtype=np.int)  # is compressed form

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


# TODO default value for derivative, change type from int to str
def generate_iso_dmat(grid: RegularGrid, derivative=4):
    """
    Computes isotropic derivative matrix.

    Parameters
    ----------
    grid : RegularGrid
    derivative : list of int

    Returns
    -------
    scipy.sparse.csc_matrix, scipy.sparse.csc_matrix
    """
    npix = grid.size
    nx = grid.nr
    # diagonal
    diam = eye(npix, npix)
    diar = spdiags((np.ones((1, npix))), 1, npix, npix, format='csc')
    # to reference right nearest neighbors
    dial = spdiags((np.ones((1, npix))), -1, npix, npix, format='csc')
    # reference left nearest neighbors
    diao = spdiags((np.ones((1, npix))), nx, npix, npix, format='csc')
    # reference upper nearest neighbors
    diau = spdiags((np.ones((1, npix))), -nx, npix, npix, format='csc')

    # selection of suitable derivative
    if derivative == 1:
        br = -diam + dial
        bz = -diam + diao
    elif derivative == 2:
        br = diam - diar
        bz = diam - diau
    elif derivative == 3:
        br = dial - 2*diam + diar
        bz = diao - 2*diam + diau
    elif derivative == 4:
        br = dial - diar
        bz = diao - diau
    br = br / grid.dr
    bz = bz / grid.dz
    return br, bz


def reduce_dmats(dmats, idx):
    """
    Creates reduced derivative matrices by cutting out rows and columns representing unwanted nodes.

    Parameters
    ----------
    dmats : tuples of scipy.sparse.csr_matrix
        tuple containing full derivative matrices
    idx : numpy.ndarray of bool
        array or mask array to select desired nodes

    Returns
    -------
    list of scipy.sparse.csr_matrix pairs
        derivative matrices with removed rows and columns representing nodes outside the bdr polygon (vacuum vessel)
    """
    dmats_red = []
    for pair in dmats:
        s1 = pair[0][idx, :][:, idx]
        s2 = pair[1][idx, :][:, idx]
        pair_red = (s1, s2)
        dmats_red.append(pair_red)
    return dmats_red


def derivative_matrix(grid: RegularGrid, direction: str, scheme: str = 'forward', mask: np.ndarray = None, compensate_edges=False):
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
        dmat = dmat[mask, :][:, mask]
    if compensate_edges:
        row_sum = np.array(dmat.sum(1)).flatten()
        row_sum_diag = sparse.diags([row_sum], [0], format='csr')
        dmat = dmat - row_sum_diag

    return dmat


def laplace_matrix(grid: RegularGrid, mask=None, compensate_edge=False, diagonals=True):
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
        lmat = lmat[mask, :][:, mask]

    if compensate_edge:
        row_sum = np.array(lmat.sum(1)).flatten()
        row_sum = sparse.diags([row_sum], [0], format='csr')
        lmat = lmat - row_sum

    return lmat


def get_index_shift(col_num: int, direction: str, distance: int = 1) -> int:
    """
    Computes the index difference of a node neighbor on a flatten grid.

    Parameters
    ----------
    col_num : int
        number of columns in reconstruction grid
    direction : str
        Determines one of the 8 possible directions that can be used.
        Specified using the over edge neighbor 'right', 'top', 'left', 'bottom' or
        combination over corner directions like 'top-left'
    distance : int, optional
        expressed in number of nodes, by default 1

    Returns
    -------
    int
        relative index of neighbor in flattened grid
    """
    if direction == 'right':
        shift = distance
    elif direction == 'top-right':
        shift = distance * (1 + col_num)
    elif direction == 'top':
        shift = col_num * distance
    elif direction == 'top-left':
        shift = distance * (-1 + col_num)
    elif direction == 'left':
        shift = - distance
    elif direction == 'bottom-left':
        shift = distance * (-1 - col_num)
    elif direction == 'bottom':
        shift = - distance * col_num
    elif direction == 'bottom-right':
        shift = distance * (1 - col_num)
    return shift
