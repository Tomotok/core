# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
"""
Handles saving and loading of sparse matrices to/from HDF files.

Currently supported formats:
 - scipy.sparse.csc_array
 - scipy.sparse.csr_array
 - scipy.sparse.dia_array
"""
import h5py
from scipy import sparse


def sparse_to_hdf(matrix: sparse.csc_array | sparse.csr_array | sparse.dia_array, group: h5py.Group):
    """
    Saves scipy.sparse matrix of formats (csc, csr, dia) into hdf file group.
    """
    if isinstance(matrix, sparse.dia_array):
        dia_to_hdf(matrix, group)
    elif isinstance(matrix, (sparse.csc_array, sparse.csr_array)):
        cs_to_hdf(matrix, group)
    else:
        raise TypeError(f'Unsupported matrix type `{type(matrix)}`. Use dia, csc or csr.')


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
        raise ValueError(f'Unsupported matrix format `{form}`.')
    return matrix


def dia_to_hdf(matrix: sparse.dia_array, group: h5py.Group):
    """
    Saves dia matrix to hdf group.
    """
    if not isinstance(matrix, sparse.dia_array):
        raise TypeError(f'Provided matrix is not of sparse diagonal type but `{type(matrix)}`.')
    group.attrs['type'] = matrix.format
    group.attrs['shape'] = matrix.shape
    group.create_dataset('offsets', data=matrix.offsets)
    group.create_dataset('data', data=matrix.data)


def hdf_to_dia(group: h5py.Group) -> sparse.dia_array:
    """
    Loads dia matrix from hdf group.
    """
    form = group.attrs['type']
    if form != 'dia':
        raise ValueError(f'Provided group attr `type` does not specify diagonal matrix but `{form}`.')
    shape = group.attrs['shape'][()]
    data = group['data'][:]
    offsets = group['offsets'][:]
    matrix = sparse.dia_array((data, offsets), shape=shape)
    return matrix


def cs_to_hdf(matrix: sparse.csc_array | sparse.csr_array, group: h5py.Group):
    """
    Saves compressed sparse matrix to hdf group.
    """
    if not isinstance(matrix, (sparse.csc_array, sparse.csr_array)):
        raise TypeError(f'Provided matrix is not of csc or csr type but {type(matrix)}.')
    group.attrs['type'] = matrix.format
    group.attrs['shape'] = matrix.shape
    group.create_dataset('indices', data=matrix.indices)
    group.create_dataset('indptr', data=matrix.indptr)
    group.create_dataset('data', data=matrix.data)


def hdf_to_cs(group: h5py.Group) -> sparse.csc_array | sparse.csr_array:
    """
    Loads compressed sparse matrix from hdf group.
    """
    form = group.attrs['type']
    if form not in ['csc', 'csr']:
        raise TypeError(f'Provided group attr `type` does not specify csc or csr matrix but `{form}`.')
    shape = group.attrs['shape'][()]
    data = group['data'][:]
    indices = group['indices'][:]
    indptr = group['indptr'][:]
    if form == 'csc':
        matrix = sparse.csc_array((data, indices, indptr), shape=shape)
    else:
        matrix = sparse.csr_array((data, indices, indptr), shape=shape)
    return matrix
