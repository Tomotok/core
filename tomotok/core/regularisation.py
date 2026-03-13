# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
import numpy as np
from scipy import sparse


def regularisation_matrix(
        derivatives: sparse.sparray | list[sparse.sparray],
        derivative_weights: float | list[float] | np.typing.ArrayLike | None = None, 
        node_weights: float | list[float] | None = None,
    ) -> sparse.csc_matrix:
    """
    Computes regularisation matrix from derivatives matrices.

    Parameters
    ----------
    derivatives : sparse.sparray or list of sparse.sparray
        sparse matrices with numerical derivative operators with shape (#nodes, #nodes)
    derivative_weights : float or list of floats, optional
        weights assigned to individual derivatives matrices
        by default all weights are equal
        if list of floats, the derivative weights are specified for each derivative matrix
        if list of arrays, the derivative weights are specified for each node
    node_weights : float or list of floats, optional
        weights assigned to individual nodes, default is 1 for all nodes
        can be used to create a non-linear regularisation matrix

    Returns
    -------
    sparse.csc_array
        regularisation matrix with shape (#nodes, #nodes)
    """
    if isinstance(derivatives, sparse.sparray):
        derivatives = [derivatives]
    derivative_shape = derivatives[0].shape
    if derivative_weights is None:
        derivative_weights = [1] * len(derivatives)
    try:
        assert len(derivative_weights) == len(derivatives)
    except AssertionError:
        raise ValueError('Derivative weights must have same length as derivatives')
    except TypeError:
        raise TypeError('Derivative weights must be a list of numbers or a list of arrays')
    if node_weights is None:
        node_weights = [1.0]

    node_weights = sparse.diags_array(node_weights, shape=derivative_shape)
    total = sum(derivative_weights)
    regularisation = sparse.csr_array(derivative_shape)
    for dw, dmat in zip(derivative_weights, derivatives):
        dw = dw / total
        if isinstance(dw, float):
            dw = [dw]
        dw_mat = sparse.diags_array(dw, shape=dmat.shape)
        regularisation += dmat.T @ (dw_mat * node_weights) @ dmat
    return regularisation
