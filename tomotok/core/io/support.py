# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Simple hdf saving and loading functions
"""
from warnings import warn

import h5py


def to_hdf(dct, floc, attrs={}):
    """
    Saves provided dict into a hdf file

    Parameters
    ----------
    dct : dict
        [description]
    floc : str
        path to file with name
    attrs : dict
        metadata to be saved to attributes
    """
    warn('Support for dense geometry matrices will be removed in the future.', FutureWarning)
    with h5py.File(floc, 'w') as fl:
        for key in dct:
            fl.create_dataset(str(key), data=dct[key])
        # if name is not None:
        for key in attrs:
            fl.attrs[key] = attrs[key]
    return


def from_hdf(floc):
    dct = {}
    with h5py.File(floc, 'r') as fl:
        for key in fl.keys():
            dct[key] = fl[key][:]
    return dct
