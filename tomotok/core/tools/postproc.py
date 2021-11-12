# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
import matplotlib.pyplot as plt
import numpy as np


def compute_power(recs, grid):
    """
    Computes total radiated power from whole reconstruction area.

    Parameters
    ----------
    grid : RegularGrid

    Returns
    -------
    numpy.ndarray
        Contains values of total radiated power for all time slices
    """
    nslices = recs.shape[0]
    pixvol = grid.nodevol.flatten()
    ptot = np.ones(nslices)
    for i in range(nslices):
        rec = recs.flatten()
        ptot[i] = rec.dot(pixvol)
    return ptot


def plot(recs, ts=0):
    """
    Moved from BaseMfr class
    """
    raise NotImplementedError('TBD')
    data = self.results['used_data']
    errors = self.results['errors']
    # chnls = data.chnl.values
    chnls = self.results['chnls']
    bcli = self.results['retrofit']
    recs = self.results['recs']

    asp = plt.figaspect(0.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=asp)

    ax1.set_title('Recs')
    # ax1.plot(*self.diag.load_boundary_coord().T, color='w')
    im = ax1.imshow(recs[ts], origin='lower', extent=self.grid.extent)
    plt.colorbar(im, ax=ax1)

    ax2.set_title('Retrofit')
    ax2.bar(chnls, bcli[ts], color='C0', alpha=1, label=r'$\mathbf{T} \cdot \mathbf{g}$')
    ax2.errorbar(chnls, data[ts, :], errors[ts, :],
                    ls='', marker='+', capsize=3, c='k', alpha=1, label='data')
    ax2.legend()
    ax2.yaxis.tick_right()
    plt.show()
    return fig