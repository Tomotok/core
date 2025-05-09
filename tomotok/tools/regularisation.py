from typing import Union, List, Optional

from scipy import sparse


Derivative_type = Union[sparse.spmatrix, List[sparse.spmatrix]]


def regularisation_matrix(
        derivatives: Derivative_type, derivative_weights=None, node_weights=None
    ) -> sparse.csc_matrix:
    """
    Computes regularisation matrix from derivatives matrices.

    Parameters
    ----------
    derivatives : sparse.spmatrix or list of sparse.spmatrix
        sparse matrices with numerical derivative operators with shape (#nodes, #nodes)
    derivative_weights : list of floats or list of array-like, optional
        weights assigned to individual derivatives matrices
        by default all weights are equal
        if list of floats, the derivative weights are specified for each derivative matrix
        if list of arrays, the derivative weights are specified for each node
    node_weights : float or list of floats, optional
        weights assigned to individual nodes, default is 1 for all nodes
        can be used to create a non-linear regularisation matrix

    Returns
    -------
    sparse.spmatrix
        regularisation matrix with shape (#nodes, #nodes)
    """
    if isinstance(derivatives, sparse.spmatrix):
        derivatives = [derivatives]
    if derivative_weights is None:
        derivative_weights = [1] * len(derivatives)
    try:
        assert len(derivative_weights) == len(derivatives)
    except AssertionError:
        raise ValueError('Derivative weights must have same length as derivatives')
    except TypeError:
        raise TypeError('Derivative weights must be a list of numbers or a list of arrays')
    if node_weights is None:
        node_weights = [1]

    node_weights = sparse.diags(node_weights, shape=derivatives[0].shape)
    total = sum(derivative_weights)
    regularisation = sparse.csr_matrix(derivatives[0].shape)
    for dw, dmat in zip(derivative_weights, derivatives):
        dw_mat = sparse.diags(dw/total, shape=dmat.shape)
        regularisation += dmat.T @ (dw_mat * node_weights) @ dmat
    return regularisation
