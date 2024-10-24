# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Simple hdf saving and loading functions
"""
from pathlib import Path
from typing import Union
from warnings import warn

import h5py


def to_hdf(dct: dict, floc: Union[str, Path], attrs: dict = {}, auto_rename=True) -> None:
    """
    Saves provided dict into a hdf file

    Parameters
    ----------
    dct : dict
        [description]
    floc : str or pathlib.Path
        path to file with name
    attrs : dict
        metadata to be saved to attributes
    """
    floc = Path(floc)
    floc = floc.expanduser()
    if floc.exists():
        if auto_rename:
            i = 0
            new = floc
            while new.exists():
                i += 1
                # FIXME: is there a missing dot before suffix? Handle no suffix case.
                new = floc.parent / f'{floc.stem}_{i}{floc.suffix}'
            warn(f'File {floc} already exists. Renaming to {new}')
            floc = new
        else:
            raise FileExistsError(f'File {floc} already exists.')
    with h5py.File(floc, 'w') as fl:
        for key in dct:
            fl.create_dataset(str(key), data=dct[key])
        # if name is not None:
        for key in attrs:
            fl.attrs[key] = attrs[key]
    return


def from_hdf(floc: Union[str, Path]) -> dict:
    """
    Loads all datasets from hdf file to dict

    Parameters
    ----------
    floc : str or pathlib.Path
        path to file with name

    Returns
    -------
    dict
        holds all datasets from hdf file
    """
    dct = {}
    floc = Path(floc)
    floc = floc.expanduser()
    with h5py.File(floc, 'r') as fl:
        for key in fl.keys():
            dct[key] = fl[key][:]
    return dct
