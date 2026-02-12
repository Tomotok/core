# Copyright 2026 Institute of Plasma Physics of the Czech Academy of Sciences. 
# Licensed under the EUPL-1.2 or later.
from numpy.typing import ArrayLike

from tomotok.tools.containers import Signals


class Diagnostic(object):
    """
    Base class for diagnostics. Should be subclassed for specific diagnostics.
    """
    def __init__(self):
        """
        Parameters
        ----------
        shot : int, optional
            Shot number for which the diagnostic instance is created. It is used as default.
        kwargs : dict
            Additional keyword arguments for subclass
        """
        return

    def load_data(self, shot: int, tvec: ArrayLike | None = None) -> Signals:
        """
        Loads diagnostic data from database specified by `source`.

        Parameters
        ----------
        shot : int
            Shot number for which the diagnostic data is loaded.
        tvec : ArrayLike, optional
            specifies timeslices to be loaded, if not provided, all available timeslices are loaded
        source : str, optional
            specifies source for loading diagnostic data

        Returns
        -------
        Signals
            Loaded diagnostics signal stored in Signals container
        """
        raise NotImplementedError('Method load_data should be defined in subclass.')
