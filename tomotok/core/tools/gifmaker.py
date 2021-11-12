# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Simple function used to generate gifs from a set of png images.

Requires imageio library. To be replaced by matplotlib's animation
"""
import os

import imageio


def gifmaker(shot, what, duration, loops, fldr='../tomography/Graphs/all/'):
    """

    Parameters
    ----------
    shot : int
        checks file name if it contains required shot number
    what : str
        a string required to be present in file name, e.g. 'rectangles'
    duration
    loops : int
    fldr : str
        url of folder with pngs

    Returns
    -------

    """
    startdir = os.getcwd()
    os.chdir(fldr)
    slices = []
    names = []

    gif = '{} {} loops {} animation.gif'.format(shot, what, loops)
    if os.path.exists(gif):
        os.remove(gif)
#    mp4 = '{} {} video.mp4'.format(shot, what)
#    if os.path.exists(mp4):
#        os.remove(mp4)
    
    for fname in os.listdir():
        if what in fname.lower() and str(shot) in fname:        
            names.append(fname)
    names.sort()
    for name in names:
        slices.append(imageio.imread(name))
    imageio.mimwrite(gif, slices, duration=duration, loop=loops)
#    imageio.mimwrite(mp4, slices)
    os.chdir(startdir)
