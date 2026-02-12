# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
from pathlib import Path
from .derivative import compute_aniso_dmats, compute_iso_dmats
from . import geometry
from . import inversions


with open(Path(__file__).parent / 'VERSION') as version_file:
    __version__ = version_file.read().strip()
