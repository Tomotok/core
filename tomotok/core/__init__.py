# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
from os import path
from .derivative import compute_aniso_dmats, compute_iso_dmats
from . import geometry
from . import inversions
from . import io


with open(path.join(path.dirname(__file__), 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
