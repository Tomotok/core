# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Contains inversion algorithms. It is independent on the rest of the package.

Currently implemented
 - Minimum Fisher Regularisation (MFR)
 - Linear Algebraic Methods (LAME)
 - Biorthogonal Basis Decomposition (BOB)
 - Fixed parameter MFR
"""

from .bob import Bob, SparseBob, SimpleBob, CholmodBob
from .lame import SvdFastAlgebraic, GevFastAlgebraic
from .mfr import Mfr, CholmodMfr
from .fixed import Fixt, CholmodFixt
