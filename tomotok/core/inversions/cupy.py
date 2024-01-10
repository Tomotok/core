# Copyright 2024 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
from typing import Union
from numpy.typing import ArrayLike

from scipy.sparse import spmatrix

from .mfr import Mfr


class Cupy(object):
    def __init__(self):
        super().__init__()
        import cupy as cp
        self.cp = cp
        
    def invert(self, a, b):
        r"""
        Finds solution of :math:`\mathbf{Ax}=\mathbf{b}` using cupy

        Parameters
        ----------
        a : array_like
            square and positive definite matrix
        b : array_like
            right hand side vector
        """
        if isinstance(a, spmatrix):
            a = a.toarray()
        cp = self.cp
        ca = cp.asarray(a)
        cb = cp.asarray(b)
        x = cp.linalg.solve(ca, cb)
        return cp.asnumpy(x)


class SparseCupy(object):
    def __init__(self):
        super().__init__()
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix
        from cupyx.scipy.sparse.linalg import spsolve
        self.spsolve = spsolve
        self.csr_matrix = csr_matrix
        self.cp = cp

    def invert(self, a, b):
        r"""
        Finds solution of :math:`\mathbf{Ax}=\mathbf{b}` using cupyx spsolve

        Parameters
        ----------
        a : array_like
            square and positive definite matrix
        b : array_like
            right hand side vector
        """
        ca = self.csr_matrix(a)
        cb = self.cp.asarray(b)
        x = self.spsolve(ca, cb)
        return self.cp.asnumpy(x)


class CupyMfr(Cupy, Mfr):
    pass


class SparseCupyMfr(SparseCupy, Mfr):
    pass
