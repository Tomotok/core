# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
import matplotlib.pyplot as plt
import numpy as np

# TODO remove?
def projection_space(chords, centre=(0, 0), magnetics=None, maglvl=1):
    """

    Parameters
    ----------
    chords : tuple
    centre : tuple of float
    magnetics : dict
    maglvl : float, list of float

    Returns
    -------
    fig : matplotlib.figure
    """
    fig, ax = plt.subplots()
    xch, ych = chords
    dr = np.diff(xch)
    dz = np.diff(ych)
    nrm = np.sqrt(np.square(dr) + np.square(dz))
    p = np.zeros_like(xch)
    n = np.zeros_like(xch)
    p[:, 0] = centre[0] - xch[:, 0]
    p[:, 1] = centre[1] - ych[:, 0]
    n[:, 0] = -dz.flatten()
    n[:, 1] = dr.flatten()
    d = (n * p).sum(axis=1) / nrm.flatten()
    # d = np.abs(d)
    xi = np.arctan2(dz, dr).flatten()
    # d = np.where(xi > 0, d, -d)
    xi = np.where(xi > 0, xi, np.pi + xi)
    ax.plot(xi, d, ls='', marker='+', label='chords')
    if magnetics is not None:
        style = 'dotted'
        clr = 'k'
        vals = magnetics['values']
        x = magnetics['x']
        y = magnetics['y']
        cnt = plt.contour(x, y, vals, levels=maglvl)
        segs = cnt.allsegs
        for coll in cnt.collections:
            coll.remove()
        tcnt = []
        for seg in segs:
            if seg == []:
                tcnt.append(seg)
            else:
                # TODO fix for contours created by multiple segments of lines
                seg = seg[0]
                dr = seg[:, 0] - centre[0]
                dz = seg[:, 1] - centre[1]
                r = np.sqrt(np.square(dr) + np.square(dz))
                mxi = np.arctan2(dz, dr)
                rnum = -mxi.argmax()  # roll to avoid jump in xi angle
                mxi = np.roll(mxi, rnum)
                r = np.roll(r, rnum)
                uidx = np.where(mxi >= 0)[0]
                lidx = np.where(mxi < 0)[0]
                ax.plot(mxi[uidx], r[uidx], color=clr, ls=style)
                ax.plot(np.pi + mxi[lidx], -r[lidx], color=clr, ls=style)
                tcrd = np.stack((mxi, r), axis=1)
                tcnt.append(tcrd)
        ax.plot([], color=clr, ls=style, label='magnetics')

    ax.set_xlabel(r'Angle $ \xi $ [rad]')
    ax.set_ylabel('Length p [m]')
    ax.legend()
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # ax.set_xlim([0, 180])
    return fig
