# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
import numpy as np
from numpy.typing import ArrayLike

from tomotok.tools.containers import Magnetics


class Tokamak(object):
    """
    A Template class that should be subclassed for specific tokamaks.

    Currently only defines a template method for loading magnetic field data for subclasses and stores divertor area.

    Attributes
    ----------
    divertor_area_r : array-like, optional
        radial coordinates of divertor area boundary polygon
    divertor_area_z : array-like, optional
        vertical coordinates of divertor area boundary polygon
    """
    def __init__(self, divertor_area_r: ArrayLike | None = None, divertor_area_z: ArrayLike | None = None):
        """
        Parameters
        ----------
        divertor_area_r : array-like, optional
            radial coordinates of divertor area boundary polygon
        divertor_area_z : array-like, optional
            vertical coordinates of divertor area boundary polygon
        """
        if any([divertor_area_r is None, divertor_area_z is None]):
            if not all([divertor_area_r is None, divertor_area_z is None]):
                raise ValueError('Both divertor_area_r and divertor_area_z should be provided or both should be None.')
        else:
            divertor_area_r = np.asarray(divertor_area_r)
            divertor_area_z = np.asarray(divertor_area_z)
            if divertor_area_r.shape != divertor_area_z.shape:
                raise ValueError('divertor_area_r and divertor_area_z should have the same shape.')
        self.divertor_area_r = divertor_area_r
        self.divertor_area_z = divertor_area_z
        return

    def load_magnetic_field(self, shot: int, tvec: ArrayLike = None) -> Magnetics:
        """
        Loads magnetic field data from database for given shot and time slices.

        Specific implementation should be provided in subclass.


        Parameters
        ----------
        shot : int
            Shot number for which the magnetic field is loaded.
        tvec : array-like, optional
            specifies vector of time slices to be loaded, if not provided, all available time slices are loaded

        Returns
        -------
        Magnetics
            Psi normalized stored in Magnetics dataclass container
        """
        raise NotImplementedError('Method load_magnetic_field should be defined in subclass.')