# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
from pathlib import Path

from . import geometry
from . import inversions
from . import regularisations


__all__ = [
    "geometry",
    "inversions",
    "regularisations",
]

with open(Path(__file__).parent / 'VERSION') as version_file:
    __version__ = version_file.read().strip()
