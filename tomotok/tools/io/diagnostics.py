# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
from numpy.typing import ArrayLike

from tomotok.tools.containers import Signals


class Diagnostic:
    """
    Base class for diagnostics. Should be subclassed for specific diagnostics.
    """
    def __init__(self):
        return

    def load_data(self, shot: int, tvec: ArrayLike | None = None) -> Signals:
        """
        Loads diagnostic data from a database. Specific implementation should be provided in subclass.

        Parameters
        ----------
        shot : int
            Shot number for which the diagnostic data is loaded.
        tvec : ArrayLike, optional
            specifies timeslices to be loaded, if not provided, all available timeslices are loaded

        Returns
        -------
        Signals
            Loaded diagnostics signal stored in Signals container
        """
        raise NotImplementedError('Method load_data should be defined in subclass.')
