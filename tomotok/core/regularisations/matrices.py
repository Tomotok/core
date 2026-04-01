# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
from scipy import sparse


def weighted_squares(
    matrices: sparse.sparray | list[sparse.sparray],
    matrix_weights: float | list[float] | None = None, 
    node_weights: float | list[float] | None = None,
) -> sparse.csc_matrix:
    """Computes regularisation matrix as a weighted sum of squares of the input matrices.

    The weights can be based on intermediate results creating a non-linear regularisation matrix.
    Proper 

    Parameters
    ----------
    matrices : sparse.sparray or list of sparse.sparray
        sparse matrices with numerical derivative operators with shape (#nodes, #nodes)
    matrix_weights : float or list of floats, optional
        weights assigned to individual matrices
        by default all weights are equal
        if list of floats, the matrix weights are specified for each matrix
        if list of arrays, the matrix weights are specified for each node of each matrix
    node_weights : float or list of floats, optional
        weights assigned to individual nodes, default is 1 for all nodes
        can be used to create a non-linear regularisation matrix

    Returns
    -------
    sparse.csc_array
        regularisation matrix with shape (#nodes, #nodes)
    """
    if isinstance(matrices, sparse.sparray):
        matrices = [matrices]
    derivative_shape = matrices[0].shape
    if matrix_weights is None:
        matrix_weights = [1] * len(matrices)
    try:
        assert len(matrix_weights) == len(matrices)
    except AssertionError:
        raise ValueError('Matrix weights must have same length as matrices')
    except TypeError:
        raise TypeError('Matrix weights must be a list of numbers or a list of arrays')
    if node_weights is None:
        node_weights = [1.0]

    node_weights = sparse.diags_array(node_weights, shape=derivative_shape)
    total = sum(matrix_weights)
    out = sparse.csr_array(derivative_shape)
    for w, mat in zip(matrix_weights, matrices):
        w = w / total
        if isinstance(w, float):
            w = [w]
        dw_mat = sparse.diags_array(w, shape=mat.shape)
        out += mat.T @ (dw_mat * node_weights) @ mat
    return out
