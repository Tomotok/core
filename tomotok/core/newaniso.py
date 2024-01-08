import numpy as np
from scipy import sparse

from tomotok.core.derivative import reduce_matrix
from tomotok.core.geometry import RegularGrid


def anisotropic(grid: RegularGrid, flux: np.ndarray, num: int = 1):
    num_col = grid.nr
    num_row = grid.nz
    horizontal = grid.dr
    vertical = grid.dz
    diagonal = (horizontal**2 + vertical**2)**0.5
    # flattened index in a submatrix transformed to flattened index in full matrix
    dirdict = {
        0: -num_col-1,  # lower left
        1: -num_col,  # lower
        2: -num_col+1,    # lower right
        3: -1,         # left
        4: 0,          # center
        5: 1,          # right
        6: num_col-1, # upper left
        7: num_col, # upper
        8: num_col+1,   # upper right
    }
    # # extend flux matrix for easier edges handling
    # mltp_edge = 2
    # mltp_corner = 4
    # flux_ext = np.zeros((num_row+2, num_col+2))
    # flux_ext[1:-1, 1:-1] = flux
    # flux_ext[0, 1:-1] = flux[0, :] * mltp_edge
    # flux_ext[-1, 1:-1] = flux[-1, :] * mltp_edge
    # flux_ext[1:-1, 0] = flux[:, 0] * mltp_edge
    # flux_ext[1:-1, -1] = flux[:, -1] * mltp_edge
    # flux_ext[0, 0] = flux[0, 0] * mltp_corner
    # flux_ext[0, -1] = flux[0, -1] * mltp_corner
    # flux_ext[-1, 0] = flux[-1, 0] * mltp_corner
    # flux_ext[-1, -1] = flux[-1, -1] * mltp_corner

    parallel_fwd = sparse.lil_matrix((grid.size, grid.size))
    parallel_bwd = sparse.lil_matrix((grid.size, grid.size))
    perpendicular_fwd = sparse.lil_matrix((grid.size, grid.size))
    perpendicular_bwd = sparse.lil_matrix((grid.size, grid.size))
    for i in range(1, num_row-1):  # row index in extended full matrix
        for j in range(1, num_col-1):  # column index in extended full matrix
            submatrix = flux[i-1:i+2,j-1:j+2]
            center = flux[i, j]
            diff = np.abs(submatrix - center)
            order = np.argsort(diff.flatten())
            idx_cen = i * num_col + j  # flattened index in full matrix
            idx_par = order[num]
            if idx_par == 4:
                idx_par = order[0]
            norm_par = select_norm(idx_par, horizontal, vertical, diagonal)
            parallel_fwd[idx_cen, idx_cen + dirdict[idx_par]] = 1 / norm_par
            parallel_fwd[idx_cen, idx_cen] = -1 / norm_par
            perpendicular_fwd[idx_cen, idx_cen + dirdict[idx_par]] = 1 / norm_par
            perpendicular_fwd[idx_cen, idx_cen] = -1 / norm_par
            idx_per = order[-num]
            norm_per = select_norm(idx_per, horizontal, vertical, diagonal)
            parallel_bwd[idx_cen, idx_cen + dirdict[idx_per]] = 1 / norm_per
            parallel_bwd[idx_cen, idx_cen] = -1 / norm_per
            perpendicular_bwd[idx_cen, idx_cen + dirdict[idx_per]] = 1 / norm_per
            perpendicular_bwd[idx_cen, idx_cen] = -1 / norm_per
    
    # bottom left corner
    parallel_fwd[0, 0] = -1 / vertical - 1 / horizontal
    parallel_fwd[0, 1] = 1 / horizontal
    parallel_fwd[0, num_col] = 1 / vertical
    perpendicular_fwd[0, 0] = -1 / diagonal
    perpendicular_fwd[0, num_col + 1] = 1 / diagonal
    # bottom right corner
    parallel_fwd[num_col-1, num_col-1] = -1 / vertical - 1 / horizontal
    parallel_fwd[num_col-1, num_col-2] = 1 / horizontal
    parallel_fwd[num_col-1, 2*num_col-1] = 1 / vertical
    perpendicular_fwd[num_col-1, num_col-1] = -1 / diagonal
    perpendicular_fwd[num_col-1, 2*num_col-2] = 1 / diagonal
    # upper left corner
    parallel_fwd[-num_col, -num_col] = -1 / vertical - 1 / horizontal
    parallel_fwd[-num_col, -num_col+1] = 1 / horizontal
    parallel_fwd[-num_col, -2*num_col] = 1 / vertical
    perpendicular_fwd[-num_col, -num_col] = -1 / diagonal
    perpendicular_fwd[-num_col, -2*num_col+1] = 1 / diagonal
    # upper right corner
    parallel_fwd[-1, -1] = -1 / vertical - 1 / horizontal
    parallel_fwd[-1, -2] = 1 / horizontal
    parallel_fwd[-1, -num_col-1] = 1 / vertical
    perpendicular_fwd[-1, -1] = -1 / diagonal
    perpendicular_fwd[-1, -num_col-2] = 1 / diagonal

    
    # bottom left corner
    parallel_bwd[0, 0] = -1 / vertical - 1 / horizontal
    parallel_bwd[0, 1] = 1 / horizontal
    parallel_bwd[0, num_col] = 1 / vertical
    perpendicular_bwd[0, 0] = -1 / diagonal
    perpendicular_bwd[0, num_col + 1] = 1 / diagonal
    # bottom right corner
    parallel_bwd[num_col-1, num_col-1] = -1 / vertical - 1 / horizontal
    parallel_bwd[num_col-1, num_col-2] = 1 / horizontal
    parallel_bwd[num_col-1, 2*num_col-1] = 1 / vertical
    perpendicular_bwd[num_col-1, num_col-1] = -1 / diagonal
    perpendicular_bwd[num_col-1, 2*num_col-2] = 1 / diagonal
    # upper left corner
    parallel_bwd[-num_col, -num_col] = -1 / vertical - 1 / horizontal
    parallel_bwd[-num_col, -num_col+1] = 1 / horizontal
    parallel_bwd[-num_col, -2*num_col] = 1 / vertical
    perpendicular_bwd[-num_col, -num_col] = -1 / diagonal
    perpendicular_bwd[-num_col, -2*num_col+1] = 1 / diagonal
    # upper right corner
    parallel_bwd[-1, -1] = -1 / vertical - 1 / horizontal
    parallel_bwd[-1, -2] = 1 / horizontal
    parallel_bwd[-1, -num_col-1] = 1 / vertical
    perpendicular_bwd[-1, -1] = -1 / diagonal
    perpendicular_bwd[-1, -num_col-2] = 1 / diagonal

    # edges
    for j in range(1, num_col-1):
        # bottom
        idx_cen = 0 * num_col + j
        idx_par = idx_cen + 1
        idx_per = idx_cen + num_col
        parallel_fwd[idx_cen, idx_par] = 1 / horizontal
        parallel_fwd[idx_cen, idx_cen] = -1 / horizontal
        perpendicular_fwd[idx_cen, idx_per] = 1 / vertical
        perpendicular_fwd[idx_cen, idx_cen] = -1 / vertical
        # top
        idx_cen = (num_row-1) * num_col + j
        idx_par = idx_cen - 1
        idx_per = idx_cen - num_col
        parallel_fwd[idx_cen, idx_cen] = -1 / horizontal
        parallel_fwd[idx_cen, idx_par] = 1 / horizontal
        perpendicular_fwd[idx_cen, idx_cen] = -1 / vertical
        perpendicular_fwd[idx_cen, idx_per] = 1 / vertical
    for i in range(1, num_row-1):
        # left
        idx_cen = i * num_col + 0
        idx_par = idx_cen + num_col
        idx_per = idx_cen + 1
        parallel_fwd[idx_cen, idx_cen] = -1 / vertical
        parallel_fwd[idx_cen, idx_par] = 1 / vertical
        perpendicular_fwd[idx_cen, idx_cen] = -1 / horizontal
        perpendicular_fwd[idx_cen, idx_per] = 1 / horizontal
        # right
        idx_cen = i * num_col + num_col-1
        idx_par = idx_cen - num_col
        idx_per = idx_cen - 1
        parallel_fwd[idx_cen, idx_cen] = -1 / vertical
        parallel_fwd[idx_cen, idx_par] = 1 / vertical
        perpendicular_fwd[idx_cen, idx_cen] = -1 / horizontal
        perpendicular_fwd[idx_cen, idx_per] = 1 / horizontal
    
    for j in range(1, num_col-1):
        # bottom
        idx_cen = 0 * num_col + j
        idx_par = idx_cen + 1
        idx_per = idx_cen + num_col
        parallel_bwd[idx_cen, idx_par] = 1 / horizontal
        parallel_bwd[idx_cen, idx_cen] = -1 / horizontal
        perpendicular_bwd[idx_cen, idx_per] = 1 / vertical
        perpendicular_bwd[idx_cen, idx_cen] = -1 / vertical
        # top
        idx_cen = (num_row-1) * num_col + j
        idx_par = idx_cen - 1
        idx_per = idx_cen - num_col
        parallel_bwd[idx_cen, idx_cen] = -1 / horizontal
        parallel_bwd[idx_cen, idx_par] = 1 / horizontal
        perpendicular_bwd[idx_cen, idx_cen] = -1 / vertical
        perpendicular_bwd[idx_cen, idx_per] = 1 / vertical
    for i in range(1, num_row-1):
        # left
        idx_cen = i * num_col + 0
        idx_par = idx_cen + num_col
        idx_per = idx_cen + 1
        parallel_bwd[idx_cen, idx_cen] = -1 / vertical
        parallel_bwd[idx_cen, idx_par] = 1 / vertical
        perpendicular_bwd[idx_cen, idx_cen] = -1 / horizontal
        perpendicular_bwd[idx_cen, idx_per] = 1 / horizontal
        # right
        idx_cen = i * num_col + num_col-1
        idx_par = idx_cen + num_col
        idx_per = idx_cen - 1
        parallel_bwd[idx_cen, idx_cen] = -1 / vertical
        parallel_bwd[idx_cen, idx_par] = 1 / vertical
        perpendicular_bwd[idx_cen, idx_cen] = -1 / horizontal
        perpendicular_bwd[idx_cen, idx_per] = 1 / horizontal

    parallel_fwd = parallel_fwd.tocsr()
    perpendicular_fwd = perpendicular_fwd.tocsr()
    return parallel_fwd, perpendicular_fwd, parallel_bwd, perpendicular_bwd


def anisotropic2(grid: RegularGrid, flux: np.ndarray, num: int = 1):
    num_col = grid.nr
    num_row = grid.nz
    horizontal = grid.dr
    vertical = grid.dz
    diagonal = (horizontal**2 + vertical**2)**0.5
    # flattened index in a submatrix transformed to flattened index in full matrix
    dirdict = {
        0: -num_col-1,  # lower left
        1: -num_col,  # lower
        2: -num_col+1,    # lower right
        3: -1,         # left
        4: 0,          # center
        5: 1,          # right
        6: num_col-1, # upper left
        7: num_col, # upper
        8: num_col+1,   # upper right
    }
    # extend flux matrix for easier edges handling
    mltp_edge = 1.01
    mltp_corner = 1.01
    flux_ext = np.zeros((num_row+2, num_col+2))
    flux_ext[1:-1, 1:-1] = flux
    flux_ext[0, 1:-1] = flux[0, :] * mltp_edge
    flux_ext[-1, 1:-1] = flux[-1, :] * mltp_edge
    flux_ext[1:-1, 0] = flux[:, 0] * mltp_edge
    flux_ext[1:-1, -1] = flux[:, -1] * mltp_edge
    flux_ext[0, 0] = flux[0, 0] * mltp_corner
    flux_ext[0, -1] = flux[0, -1] * mltp_corner
    flux_ext[-1, 0] = flux[-1, 0] * mltp_corner
    flux_ext[-1, -1] = flux[-1, -1] * mltp_corner

    parallel_fwd = sparse.lil_matrix((flux_ext.size, flux_ext.size))
    parallel_bwd = sparse.lil_matrix((flux_ext.size, flux_ext.size))
    perpendicular_fwd = sparse.lil_matrix((flux_ext.size, flux_ext.size))
    perpendicular_bwd = sparse.lil_matrix((flux_ext.size, flux_ext.size))
    for i in range(1, num_row+1): 
        for j in range(1, num_col+1):
            submatrix = flux_ext[i-1:i+2,j-1:j+2]
            center = flux_ext[i, j]
            diff = np.abs(submatrix - center)
            order = np.argsort(diff.flatten())
            idx_cen = i * num_col + j  # flattened index in full matrix
            idx_par = order[num]
            if idx_par == 4:
                idx_par = order[0]
            norm_par = select_norm(idx_par, horizontal, vertical, diagonal)
            parallel_fwd[idx_cen, idx_cen + dirdict[idx_par]] = 1 / norm_par
            parallel_fwd[idx_cen, idx_cen] = -1 / norm_par
            parallel_bwd[idx_cen, idx_cen - dirdict[idx_par]] = 1 / norm_par
            parallel_bwd[idx_cen, idx_cen] = -1 / norm_par
            idx_per = order[-num]
            norm_per = select_norm(idx_per, horizontal, vertical, diagonal)
            perpendicular_fwd[idx_cen, idx_cen + dirdict[idx_per]] = 1 / norm_per
            perpendicular_fwd[idx_cen, idx_cen] = -1 / norm_per
            perpendicular_bwd[idx_cen, idx_cen - dirdict[idx_per]] = 1 / norm_per
            perpendicular_bwd[idx_cen, idx_cen] = -1 / norm_per
    
    parallel_fwd = parallel_fwd.tocsr()
    parallel_bwd = parallel_bwd.tocsr()
    perpendicular_fwd = perpendicular_fwd.tocsr()
    perpendicular_bwd = perpendicular_bwd.tocsr()
    
    mask_ext = np.ones(flux_ext.shape, dtype=bool)
    mask_ext[0, :] = False
    mask_ext[-1, :] = False
    mask_ext[:, 0] = False
    mask_ext[:, -1] = False
    mask_ext = mask_ext.flatten()

    parallel_fwd = reduce_matrix(parallel_fwd, mask_ext)
    parallel_bwd = reduce_matrix(parallel_bwd, mask_ext)
    perpendicular_fwd = reduce_matrix(perpendicular_fwd, mask_ext)
    perpendicular_bwd = reduce_matrix(perpendicular_bwd, mask_ext)
    
    return parallel_fwd, perpendicular_fwd, parallel_bwd, perpendicular_bwd


def select_norm(idx: int, horizontal: float, vertical: float, diagonal: float):
    """Uses local flat index to select norm for derivative matrix."""
    if idx not in range(9):
        raise ValueError(f'Index {idx} is not a valid local flat index.')
    if idx == 4:
        raise ValueError('Index pointing to center of submatrix can not be used for normalisation.')
    if idx in [3, 5]:
        return horizontal
    elif idx in [1, 7]:
        return vertical
    else:
        return diagonal


def anisotropic3(grid: RegularGrid, ipsi: np.ndarray, mask=None, scheme='forward'):
    # get flux surface tangent directions
    grad = np.gradient(ipsi)
    atan2 = np.arctan2(grad[0], grad[1])
    # shift arctan2 zero to have segments centered along directions and divide in 2 pi / 8 segments
    atan_mod = (atan2+np.pi/8) // (np.pi / 4)
    directions_per = atan_mod.astype(int).flatten()
    directions_par = ((atan_mod+2)%8).astype(int).flatten()
    if scheme=='backward':
        directions_per = (directions_per + 4) % 8
        directions_par = (directions_par + 4) % 8

    diagonals_par = np.zeros((grid.size, 9))
    diagonals_per = np.zeros((grid.size, 9))
    # transition from range(8) to a direction
    # counter clockwise direction starting with right
    # atan_to_flat = {0: 5, 1: 8, 2: 7, 3: 6, 4: 3, 5: 0, 6: 1, 7: 2}
    dirlist = np.array((5, 8, 7, 6, 3, 0, 1, 2), dtype=int)
    # 6 7 8
    # 3 4 5
    # 0 1 2 
    # this might need a for loop
    idx_per = dirlist[atan_mod.astype(int)]
    idx_par = dirlist[((atan_mod+2)%8).astype(int)]
    # diagonals_per[:, idx_per] = 1

    # alternatively?
    for i in range(9):
        diagonals_per[idx_per==i, i] = 1
        diagonals_par[idx_par==i, i] = 1
    
    horizontal = grid.dr
    vertical = grid.dz
    diagonal = (horizontal**2 + vertical**2)**0.5
    # normalize
    # lb, rb, lu, ru
    diagonals_par[:, [0, 2, 6, 8]] /= diagonal
    diagonals_per[:, [0, 2, 6, 8]] /= diagonal
    # l, r
    diagonals_par[:, [3, 5]] /= horizontal  
    diagonals_per[:, [3, 5]] /= horizontal
    # b, u
    diagonals_par[:, [1, 7]] /= vertical 
    diagonals_per[:, [1, 7]] /= vertical
    # center
    sum_par = diagonals_par.sum(axis=1)
    diagonals_par[:, 4] = -sum_par
    sum_per = diagonals_per.sum(axis=1)
    diagonals_per[:, 4] = -sum_per

    par = sparse.diags(diagonals_par.T, offsets=(-grid.nr-1, -grid.nr, -grid.nr+1, -1, 0, 1, grid.nr-1, grid.nr, grid.nr+1), format='csr')
    per = sparse.diags(diagonals_per.T, offsets=(-grid.nr-1, -grid.nr, -grid.nr+1, -1, 0, 1, grid.nr-1, grid.nr, grid.nr+1), format='csr')

    return par, per
