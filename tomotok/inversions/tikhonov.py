# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences.
# Licensed under the EUPL-1.2 or later.
import numpy as np

from .base import RegularisedInversion


class Tikhonov(RegularisedInversion):
    """Implements inversion based on Phillips-Tikhonov regularisation scheme."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gdg = None
        self._gdsig = None

    def _cache_inputs(self, data, gmat, regularisation):
        self._gdg = gmat.T @ gmat
        self._gdsig = gmat.T @ data
        super()._cache_inputs(data, gmat, regularisation)

    def invert(self, alpha: float) -> np.ndarray:
        r"""Inverts the regularised problem using provided regularisation parameter.

        Uses modified problem formulation and cached inputs to solve the following equation for :math:`\mathbf{g}`:
        
        .. math::
            (\mathbf{T}^T \dcot \mathbf{T} + \alpha \mathbf{H}) \mathbf{g} = \mathbf{G}^T \mathbf{f}  

        Parameters
        ----------
        alpha : float
            regularisation parameter
        """
        mod_mat = self._gdg + alpha * self._regularisation
        g = self.solver.solve(mod_mat, self._gdsig)
        return g
