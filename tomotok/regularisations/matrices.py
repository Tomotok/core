# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse


def weighted_squares(
    matrices: sparse.sparray | list[sparse.sparray],
    matrix_weights: list[float] | list[np.ndarray] | None = None,
    node_weights: ArrayLike | None = None,
) -> sparse.csr_array:
    """
    Computes regularisation matrix as a weighted sum of squares of the input matrices.

    The node weights can be based on a previous result, creating an iterative nonlinear regularisation scheme.
    For example, the MFR algorithm uses the inverse of the solution from the previous iteration as node weights.

    Parameters
    ----------
    matrices : scipy.sparse.sparray or list of scipy.sparse.sparray
        sparse matrices with numerical derivative operators with shape (#nodes, #nodes)
    matrix_weights : list of float or list of numpy.ndarray, optional
        weights of individual matrices, i.e. anisotropy coefficients, normalised to unit sum,
        a list of arrays with shape (#nodes,) specifies the weight of each matrix in each node
        (row of the matrix), the weights are then normalised to unit sum in each node separately,
        by default all weights are equal
    node_weights : array-like, optional
        weights of individual nodes with shape (#nodes,), by default 1 for all nodes,
        can be used to create a nonlinear regularisation matrix

    Returns
    -------
    scipy.sparse.csr_array
        regularisation matrix with shape (#nodes, #nodes)

    Raises
    ------
    ValueError
        if the number of matrix weights differs from the number of matrices
    TypeError
        if matrix weights are not a list
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
