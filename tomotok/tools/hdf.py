# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
"""
Handles saving and loading of sparse matrices to/from HDF files.

Currently supported formats:
 - scipy.sparse.csc_array
 - scipy.sparse.csr_array
 - scipy.sparse.dia_array

Legacy sparse matrices (e.g. ``scipy.sparse.csc_matrix``) are not supported and have to be converted to sparse arrays.
"""
import h5py
from scipy import sparse


def sparse_to_hdf(matrix: sparse.csc_array | sparse.csr_array | sparse.dia_array, group: h5py.Group):
    """
    Save sparse array of format csc, csr or dia into hdf file group.

    Parameters
    ----------
    matrix : scipy.sparse.csc_array or scipy.sparse.csr_array or scipy.sparse.dia_array
        sparse array to be saved
    group : h5py.Group
        group of an open hdf file where the array is stored

    Raises
    ------
    TypeError
        if the matrix format is not supported
    """
    if isinstance(matrix, sparse.dia_array):
        dia_to_hdf(matrix, group)
    elif isinstance(matrix, (sparse.csc_array, sparse.csr_array)):
        cs_to_hdf(matrix, group)
    else:
        raise TypeError(f'Unsupported matrix type `{type(matrix)}`. Use dia, csc or csr.')


def hdf_to_sparse(group: h5py.Group) -> sparse.sparray:
    """
    Load sparse array of format csc, csr or dia from hdf file group.

    Parameters
    ----------
    group : h5py.Group
        group of an open hdf file containing the array saved by :func:`sparse_to_hdf`

    Returns
    -------
    scipy.sparse.csc_array or scipy.sparse.csr_array or scipy.sparse.dia_array
        loaded sparse array

    Raises
    ------
    ValueError
        if the stored format is not supported
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
    Save sparse diagonal array to hdf group.

    Parameters
    ----------
    matrix : scipy.sparse.dia_array
        sparse array to be saved
    group : h5py.Group
        group of an open hdf file where the array is stored
    """
    if not isinstance(matrix, sparse.dia_array):
        raise TypeError(f'Provided matrix is not of sparse diagonal type but `{type(matrix)}`.')
    group.attrs['type'] = matrix.format
    group.attrs['shape'] = matrix.shape
    group.create_dataset('offsets', data=matrix.offsets)
    group.create_dataset('data', data=matrix.data)


def hdf_to_dia(group: h5py.Group) -> sparse.dia_array:
    """
    Load sparse diagonal array from hdf group.

    Parameters
    ----------
    group : h5py.Group
        group of an open hdf file containing the array

    Returns
    -------
    scipy.sparse.dia_array
        loaded sparse array
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
    Save compressed sparse array to hdf group.

    Parameters
    ----------
    matrix : scipy.sparse.csc_array or scipy.sparse.csr_array
        sparse array to be saved
    group : h5py.Group
        group of an open hdf file where the array is stored
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
    Load compressed sparse array from hdf group.

    Parameters
    ----------
    group : h5py.Group
        group of an open hdf file containing the array

    Returns
    -------
    scipy.sparse.csc_array or scipy.sparse.csr_array
        loaded sparse array
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
