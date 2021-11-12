# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Contains classes and functions used for creating database interfaces.

 - base class Dsystem for a general diagnostic system
 - sublcass with magnetics handling typical for tokamaks 
 - simple hdf interface using h5py
"""
from .diag import Dsystem
from .support import to_hdf, from_hdf
from .tokamaks import Tokamak
