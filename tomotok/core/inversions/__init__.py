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

from .bob import Bob, SparseBob
from .lame import SvdFastAlgebraic, GevFastAlgebraic
from .sksparse import CholmodMfr, CholmodFixt, CholmodBob
from .mfr import Mfr
from .fixed import Fixt
from .jax import JaxedMfr, JaxedFixt, JaxedBob
