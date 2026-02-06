# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Handles saving and loading of sparse matrices to/from HDF files.

Currently supported formats:
 - scipy.sparse.csc_matrix
 - scipy.sparse.csr_matrix
 - scipy.sparse.dia_matrix
"""
from typing import Union

import h5py
import scipy.sparse as sparse


Cs_type = Union[sparse.csc_matrix, sparse.csr_matrix]
Sparse_type = Union[sparse.dia_matrix, Cs_type]


def sparse_to_hdf(matrix: Sparse_type, group: h5py.Group):
    """
    Saves scipy.sparse matrix of formats (csc, csr, dia) into hdf file group.
    """
    if isinstance(matrix, sparse.dia_matrix):
        dia_to_hdf(matrix, group)
    elif isinstance(matrix, (sparse.csc_matrix, sparse.csr_matrix)):
        cs_to_hdf(matrix, group)
    else:
        raise TypeError('Unsupported matrix type {}. Use dia, csc or csr.'.format(type(matrix)))
    return


def hdf_to_sparse(group: h5py.Group) -> sparse.spmatrix:
    """
    Loads scipy.sparse matrix of formats (csc, csr, dia) from hdf file group.
    """
    form = group.attrs['type']
    if form == 'dia':
        matrix = hdf_to_dia(group)
    elif form in ['csc', 'csr']:
        matrix = hdf_to_cs(group)
    else:
        raise ValueError('Unsupported matrix format {}.'.format(form))
    return matrix


def dia_to_hdf(matrix: sparse.dia_matrix, group: h5py.Group):
    """
    Saves dia matrix to hdf group.
    """
    if not isinstance(matrix, sparse.dia_matrix):
        raise ValueError('Provided matrix is not of sparse diagonal type.')
    group.attrs['type'] = matrix.format
    group.attrs['shape'] = matrix.shape
    group.create_dataset('offsets', data=matrix.offsets)
    group.create_dataset('data', data=matrix.data)
    return


def hdf_to_dia(group: h5py.Group) -> sparse.dia_matrix:
    """
    Loads dia matrix from hdf group.
    """
    form = group.attrs['type']
    if form != sparse.dia_matrix.format:
        raise ValueError('Provided group attr `type` does not specify diagonal matrix.')
    shape = group.attrs['shape'][()]
    data = group['data'][:]
    offsets = group['offsets'][:]
    matrix = sparse.dia_matrix((data, offsets), shape=shape)
    return matrix


def cs_to_hdf(matrix: Cs_type, group: h5py.Group):
    """
    Saves compressed sparse matrix to hdf group.
    """
    if not isinstance(matrix, (sparse.csc_matrix, sparse.csr_matrix)):
        raise ValueError('Provided matrix is not of csc or csr type.')
    group.attrs['type'] = matrix.format
    group.attrs['shape'] = matrix.shape
    group.create_dataset('indices', data=matrix.indices)
    group.create_dataset('indptr', data=matrix.indptr)
    group.create_dataset('data', data=matrix.data)
    return


def hdf_to_cs(group: h5py.Group) -> Cs_type:
    """
    Loads compressed sparse matrix from hdf group.
    """
    form = group.attrs['type']
    if form not in [sparse.csc_matrix.format, sparse.csr_matrix.format]:
        raise ValueError('Provided group attr `type` does not specify csc or csr matrix.')
    shape = group.attrs['shape'][()]
    data = group['data'][:]
    indices = group['indices'][:]
    indptr = group['indptr'][:]
    if form == 'csc':
        matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
    else:
        matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)
    return matrix
