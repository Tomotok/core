# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Region of interest analysis widgets
"""
import os
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib import patches

from tomotok.core.phantoms import iso_psi
from .widgets_tomo import GraphRgb


class GraphRectangles(GraphRgb):
    """
    Based on GraphRgb class. Usefull for observing power in specified areas.
    
    Parameters
    ----------
    parent : TKInter class parent
        Default is None.
    data : dict
        Contains data, default is empty dict.
    
    Keywords
    --------
    title : str
        Name of window.
    """

    # TODO axis limits in cm
    def __init__(self, parent=None, **kw):
        #        tk.Toplevel.__init__(self, parent)
        GraphRgb.__init__(self, parent, **kw)
        if not 'title' in kw: self.title('Rectangles')
        #        xmax = self.shape[1]
        #        ymax = self.shape[0]
        #        wmax = xmax
        #        hmax = ymax
        self.ax_orig = self.ax.get_position()
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_axis_off()

        self.pprop = ttk.Frame(self, padding=(3, 3, 3, 3))  # , relief = 'solid')
        xrow = 1
        self.pprop.grid(row=0, column=2, rowspan=2, sticky='NSWE')
        self.xvar = tk.StringVar(value=self.xmin)
        self.xsbox = tk.Spinbox(self.pprop,
                                values=tuple(self.xgrid - self.dx / 2),
                                #                                from_=0, to=xmax - 1, increment=1.,
                                textvariable=self.xvar,
                                command=self.movex, width=5)
        self.xsbox.grid(row=xrow, column=1)
        self.xlbl = ttk.Label(self.pprop, text='left')
        self.xlbl.grid(row=xrow, column=0)

        self.yvar = tk.StringVar(value=self.ymin)
        self.ysbox = tk.Spinbox(self.pprop,
                                values=tuple(self.ygrid - self.dy / 2),
                                #                                from_ = 0, to=ymax - 1, increment = 1.,
                                textvariable=self.yvar,
                                command=self.movey, width=5)
        self.ysbox.grid(row=xrow + 1, column=1)
        self.ylbl = ttk.Label(self.pprop, text='bottom')
        self.ylbl.grid(row=xrow + 1, column=0)

        self.wvar = tk.StringVar(value=self.dx)
        wvals = tuple(self.xgrid - self.xgrid[0])[:-1]
        self.wsbox = tk.Spinbox(self.pprop,
                                values=wvals,
                                #                                from_ = 1, to = wmax, increment = 1.,
                                textvariable=self.wvar,
                                command=self.movew, width=5)
        self.wsbox.grid(row=xrow + 2, column=1)
        self.wlbl = ttk.Label(self.pprop, text='width')
        self.wlbl.grid(row=xrow + 2, column=0)

        self.hvar = tk.StringVar(value=self.dy)
        hvals = tuple(self.ygrid - self.ygrid[0])[:-1]
        self.hsbox = tk.Spinbox(self.pprop,
                                values=hvals,
                                #                                from_ = 1, to = hmax, increment = 1.,
                                textvariable=self.hvar,
                                command=self.moveh, width=5)
        self.hsbox.grid(row=xrow + 3, column=1)
        self.hlbl = ttk.Label(self.pprop, text='height')
        self.hlbl.grid(row=xrow + 3, column=0)

        self.snvar = tk.StringVar(value=1)
        self.snsbox = tk.Spinbox(self.pprop, from_=1,
                                 to=min(100, len(self.vals)),
                                 increment=1,
                                 textvariable=self.snvar,
                                 width=5)
        self.snsbox.grid(row=xrow + 6, column=1)
        self.snlbl = ttk.Label(self.pprop, text='n')
        self.snlbl.grid(row=xrow + 6, column=0)
        self.snbtn = ttk.Button(self.pprop, text='Save n-th',
                                command=self.save_nth)
        self.snbtn.grid(row=xrow + 7, column=0, columnspan=2)

        self.pcount = 0
        self.pnum = tk.StringVar(value=-1)
        self.patches = []
        self.pnumlist = tk.StringVar()
        self.pnumlist.set([i + 1 for i in range(self.pcount)])
        self.pnumlbl = ttk.Label(self.pprop, text='Patch:')
        self.pnumlbl.grid(row=0, column=0)
        self.pnumcbox = ttk.Combobox(self.pprop, values=self.pnumlist,
                                     textvariable=self.pnum,
                                     state='readonly', width=5)
        self.pnumcbox.grid(row=0, column=1)

        self.addbut = ttk.Button(self.pprop, text='+',
                                 command=self.make_patch, width=4)
        self.addbut.grid(row=5, column=0)
        self.rembut = ttk.Button(self.pprop, text='-',
                                 command=self.remove_patch, width=4)
        self.rembut.grid(row=5, column=1)

        self.dobut = ttk.Button(self.pprop, text='Compute',
                                command=self.compute, width=12)
        self.dobut.grid(row=6, columnspan=2, sticky='WE')

        predef = self.get_predef()
        for i in predef:
            self.make_patch(**i)
        self.pnum.trace('w', self.pselect)
        self.pnum.set(1)

        self.style = {'bol': 'solid', 'ntr': 'dashdot', 'sxr': 'dashed'}

    def pselect(self, *args):
        """
        Changes spinbox values to selected patch when selection changes.
        """
        val = self.get_pnum()
        if val >= 0:
            x = self.ax.patches[val].get_x()
            self.xvar.set(x)
            y = self.ax.patches[val].get_y()
            self.yvar.set(y)
            w = self.ax.patches[val].get_width()
            self.wvar.set(w)
            h = self.ax.patches[val].get_height()
            self.hvar.set(h)

    def get_predef(self):
        """
        Predefined rectangles in interesting areas are defined here.
        
        Returns
        -------
        dict
            Contains coordinates of predefined rectangles
        """
        out = [
            {'x': -1, 'y': -1, 'w': -1, 'h': -1},
            #                {'x' : 25, 'y' : 37, 'w' : 15, 'h' : 30},
            #                {'x' : 40, 'y' : 37, 'w' : 10, 'h' : 30},
            # {'x' : 10, 'y' : 70, 'w' : 20, 'h' : 10},
        ]
        return out

    def get_pnum(self):
        """
        Variable pnum represents selected patch, starts from 1.
        
        Returns
        -------
        out : int
            Number of selected patch (rectangle) starting from 0. Can be used
            as idices for Axes.patches array.
        """
        out = int(self.pnum.get()) - 1
        return out

    def raise_pcount(self):
        """
        Variable pcount represents total number of patches.
        Should be called after adding a new patch.
        """
        self.pcount += 1
        val = self.pcount
        #        self.patches += [val]
        self.pnumcbox['values'] = [i + 1 for i in range(val)]

    def lower_pcount(self):
        """
        Variable pcount represents total number of patches.
        Should be called after removing a new patch.
        """
        self.pcount -= 1
        val = self.pcount
        #        self.patches += [val]
        self.pnumcbox['values'] = [i + 1 for i in range(val)]

    def make_patch(self, x=-1, y=-1, w=-1, h=-1):
        """
        Creates new patch and adds it to axes.
        """
        if x == -1:
            x = self.xmin
        if y == -1:
            y = self.ymin
        if w == -1:
            w = self.dx
        if h == -1:
            h = self.dy
        val = self.pcount
        self.ax.add_patch(patches.Rectangle((x, y), w, h, fill=0, lw=2,
                                            ec='C{}'.format(val % 10)))
        self.raise_pcount()
        self.pnum.set(val + 1)
        self.gcanvas.draw()

    def remove_patch(self):
        """
        Removes last patch from computing and from axes.
        """
        val = self.pcount - 1
        if val > 0:
            self.ax.patches[val].remove()
            self.lower_pcount()
            self.plot(self.scale.get())
            pnum = int(self.pnum.get())
            if pnum > val:
                self.pnum.set(val)

    def movex(self, *args):
        pnum = self.get_pnum()
        p = self.ax.patches[pnum]
        val = float(self.xsbox.get())
        p.set_x(val)  # +self.xmin)
        self.gcanvas.draw()

    def movey(self, *args):
        pnum = self.get_pnum()
        p = self.ax.patches[pnum]
        val = float(self.ysbox.get())
        p.set_y(val)  # +self.ymin)
        self.gcanvas.draw()

    def movew(self, *args):
        pnum = self.get_pnum()
        p = self.ax.patches[pnum]
        val = float(self.wsbox.get())
        p.set_width(val)
        self.gcanvas.draw()

    def moveh(self, *args):
        pnum = self.get_pnum()
        p = self.ax.patches[pnum]
        val = float(self.hsbox.get())
        p.set_height(val)
        self.gcanvas.draw()

    def compute(self, *args):
        style = self.style
        #        col = self.col
        self.ax.set_position([0.125, 0.125, 0.35227272727272724, 0.755])
        self.ax2.cla()
        self.ax2.set_axis_on()
        self.ax2.set_xlabel('Time [s]')
        self.ax2.set_ylabel('Pixel normalized power [-]')
        self.ax2.yaxis.tick_right()
        self.ax2.yaxis.set_label_position('right')
        count = self.pcount
        lines = {}
        for i in self.diags:
            lines[i] = []
            self.ax2.plot([], c='k', ls=style[i], label=i)
        for i in range(count):
            p = self.ax.patches[i]
            l = np.argmin(np.abs(self.xgrid - float(p.get_x())))
            b = np.argmin(np.abs(self.ygrid - float(p.get_y())))
            r = l + int(np.round(float(p.get_width()) / self.dx))
            t = b + int(np.round(float(p.get_height()) / self.dy))
            for d in self.diags:
                lines[d].append(np.sum(self.data[d][..., b:t, l:r], axis=(1, 2)))
                #                lines[d].append(np.sum(self.odata[d][b:t,l:r,...],axis = (0,1)))
                self.ax2.plot(self.tvecs[d], lines[d][i], c='C{}'.format(i), ls=style[d])
        self.ax.legend()
        self.ax2.legend()
        ind = self.scale.get()
        self.vline = self.ax2.axvline(self.vals[ind], c='k', ls='dashed', lw=1)
        y = self.ax2.get_ylim()[1] * 1.01
        x = self.vals[ind]
        self.text = self.ax2.text(x, y, str(x))
        self.gcanvas.draw()

    def save(self, fname='Graphs/'):
        if not os.path.exists(fname):
            os.makedirs(fname)
        v = self.scale.get()
        self.fig.savefig(fname +
                         '{} at {:.0f} {:.0f}-{:.0f}ms rectangles.png'.format(
                         # '{} at {:.0f} {:.0f}-{:.0f}ms rectangles.eps'.format(
                            self.shot, 1000 * self.vals[v], 1000 * self.vals[0], 1000 * self.vals[-1]))

    def update_data(self, ndata, **kw):
        GraphRgb.update_data(self, ndata, **kw)
        if self.grids[0]:
            self.xgrid = ndata[self.key]['R']
        else:
            self.xgrid = np.arange(self.shape[1])
        if self.grids[1]:
            self.ygrid = ndata[self.key]['Z']
        else:
            self.ygrid = np.arange(self.shape[0])
        if len(self.fig.axes) > 1:
            self.ax.set_position(self.ax_orig)
            self.ax2.cla()
            self.ax2.set_axis_off()
        self.ax.lines.clear()
        for i in self.diags:
            self.ax.plot([], c=self.col[i], ls='', marker='s', label=i)
        self.ax.legend()
        self.gcanvas.draw()

    def plot(self, val, *args):
        inds = self.get_rgbval(val)
        b = self.data['bol'][inds[0]]
        n = self.data['ntr'][inds[1]]
        s = self.data['sxr'][inds[2]]
        rgb = np.stack((b, n, s), axis=2)
        rgb = rgb / np.max(rgb)
        cmy = np.ones_like(rgb) - rgb
        rgb = cmy
        if self.grids[0]:
            self.ax.set_xlabel('R [m]')
        else:
            self.ax.set_xlabel('Pixel')
        if self.grids[0]:
            self.ax.set_ylabel('z [m]')
        else:
            self.ax.set_ylabel('Pixel')

        if len(self.ax.images) == 0:
            self.ax.imshow(rgb, origin='lower', extent=self.extent)
        else:
            self.ax.images[0].set_data(rgb)
        #            self.ax.images[0].set_extent(self.extent)
        for i in self.ax.collections:
            i.remove()
        if self.magvar.get():
            mval = self.get_mval(val)
            self.cont = self.ax.contour(self.mdata[mval], [1.0],
                                        extent=self.extent,
                                        # colors='w',
                                        colors='k',
                                        linewidths=1)
        if len(self.ax2.texts) > 0:
            x = self.vals[val]
            self.vline.set_xdata(x)
            self.text.set_x(x)
            self.text.set_text(str(x))
        self.gcanvas.draw()

    def save_nth(self):
        """
        Saves every nth slice, where n is defined by snvar Stringvar asociated
        with snsbox Spinbox widget.
        """
        i = 0
        n = int(self.snvar.get())
        while i < len(self.vals):
            self.go_to(i)
            self.plot(i)
            self.save(fname='Graphs/recseq/{}/'.format(self.shot))
            i += n

class GraphContours(GraphRgb):
    """
    Based on GraphRgb class. Usefull for observing power in specified areas.
    
    Parameters
    ----------
    parent : TKInter class parent
        Default is None.
    data : dict
        Contains data, default is empty dict.
    
    Keywords
    --------
    title : str
        Name of window.
    """

    def __init__(self, parent=None, **kw):
        #        tk.Toplevel.__init__(self, parent)
        GraphRgb.__init__(self, parent, **kw)
        if 'title' not in kw:
            self.title('Contours')
        self.ax_orig = self.ax.get_position()
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_axis_off()

        self.pprop = ttk.Frame(self, padding=(3, 3, 3, 3))  # , relief = 'solid')
        xrow = 1
        self.pprop.grid(row=0, column=2, rowspan=2, sticky='NSWE')
        self.minvar = tk.StringVar(value=10)
        self.minsbox = tk.Spinbox(self.pprop, from_=0., to=2,
                                  increment=0.01, textvariable=self.minvar,
                                  command=self.move_limits, width=5)
        self.minsbox.grid(row=xrow, column=1)
        self.minlbl = ttk.Label(self.pprop, text='min')
        self.minlbl.grid(row=xrow, column=0)

        self.maxvar = tk.StringVar(value=10)
        self.maxsbox = tk.Spinbox(self.pprop, from_=0., to=2,
                                  increment=0.01, textvariable=self.maxvar,
                                  command=self.move_limits, width=5)
        self.maxsbox.grid(row=xrow + 1, column=1)
        self.maxlbl = ttk.Label(self.pprop, text='max')
        self.maxlbl.grid(row=xrow + 1, column=0)

        self.snvar = tk.StringVar(value=1)
        self.snsbox = tk.Spinbox(self.pprop, from_=1,
                                 to=min(100, len(self.vals)),
                                 increment=1,
                                 textvariable=self.snvar,
                                 width=5)
        self.snsbox.grid(row=xrow + 6, column=1)
        self.snlbl = ttk.Label(self.pprop, text='n')
        self.snlbl.grid(row=xrow + 6, column=0)
        self.snbtn = ttk.Button(self.pprop, text='Save n-th',
                                command=self.save_nth)
        self.snbtn.grid(row=xrow + 7, column=0, columnspan=2)

        self.pcount = 0
        self.pnum = tk.StringVar(value=-1)
        self.patches = []
        self.pnumlist = tk.StringVar()
        self.pnumlist.set([i + 1 for i in range(self.pcount)])
        self.pnumlbl = ttk.Label(self.pprop, text='Contours:')
        self.pnumlbl.grid(row=0, column=0)
        self.pnumcbox = ttk.Combobox(self.pprop, values=self.pnumlist,
                                     textvariable=self.pnum,
                                     state='readonly', width=5)
        self.pnumcbox.grid(row=0, column=1)

        self.addbut = ttk.Button(self.pprop, text='+',
                                 command=self.make_patch, width=4)
        self.addbut.grid(row=5, column=0)
        self.rembut = ttk.Button(self.pprop, text='-',
                                 command=self.remove_patch, width=4)
        self.rembut.grid(row=5, column=1)

        self.dobut = ttk.Button(self.pprop, text='Compute',
                                command=self.compute, width=12)
        self.dobut.grid(row=6, columnspan=2, sticky='WE')

        predef = self.get_predef()
        for i in predef:
            self.make_patch(**i)
        self.pnum.trace('w', self.pselect)
        self.pnum.set(1)
        self.style = {'bol': 'solid', 'ntr': 'dashdot', 'sxr': 'dashed'}
        self.plot(0)

    def pselect(self, *args):
        """
        Changes spinbox values to selected contours when selection changes.
        """
        val = self.get_pnum()
        if val >= 0:
            min_, max_ = self.get_levels(val)
            self.minvar.set(min_)
            self.maxvar.set(max_)

    def get_levels(self, pnum):
        """
        Gets levels from contour object. If length of levels is one,
        returns array with zero as minimum.
        
        Parameters
        ---------
        pnum : int
            Number of patch
        
        Returns
        -------
        limits : np.ndarray
            Array with minimum and maximum limited by given contours.
        """
        p = self.patches[pnum]
        limits = p.levels
        if len(limits) == 1:
            limits = np.append(0, limits)
        return limits

    def get_predef(self):
        """
        Returns predefined contours. Used for initalization of wondow.
        
        Returns
        -------
        out : dict
            Contains coordinates of predefined rectangles
        """
        out = [
            {'min_': 0.5, 'max_': 0.7},
            {'min_': 0.7, 'max_': 1.0},
            # {'x' : 10, 'y' : 70, 'w' : 20, 'h' : 10},
        ]
        return out

    def get_pnum(self):
        """
        Variable pnum represents selected contours, starts from 1.
        
        Returns
        -------
        out : int
            Number of selected contours pair starting from 0.
        """
        out = int(self.pnum.get()) - 1
        return out

    def raise_pcount(self):
        """
        Variable pcount represents total number of patches.
        Should be called after adding a new patch.
        """
        self.pcount += 1
        val = self.pcount
        #        self.patches += [val]
        self.pnumcbox['values'] = [i + 1 for i in range(val)]

    def decrease_pcount(self):
        """
        Variable pcount represents total number of patches.
        Should be called after removing a new patch.
        """
        self.pcount -= 1
        val = self.pcount
        #        self.patches += [val]
        self.pnumcbox['values'] = [i + 1 for i in range(val)]

    def add_patch(self, contours):
        """
        Adds collections created by contours to axes
        """
        for i in contours.collections.copy():
            self.ax.add_collection(i)

    def clear_patch(self, contours):
        """
        Removes collections representing given cotnours from axes.
        """
        colls = contours.collections.copy()
        for i in colls:
            self.ax.collections.remove(i)

    def make_patch(self, min_=0.5, max_=1, pnum=None):
        """
        Creates new contours and adds collections to axes.
        """
        if pnum == None:
            val = self.pcount
        else:
            val = pnum
        if float(min_) == 0.:
            ls = 'solid'
        #            lw = 3
        else:
            #            lw = [1, 3]
            #            ls = ['dotted','solid']
            ls = ['dashdot', 'solid']
        mval = self.get_mval(self.scale.get())
        cnt = self.ax.contour(self.mdata[mval], [min_, max_],
                              extent=self.extent,
                              colors=['C{}'.format(val % 10)],
                              #                              linewidths=lw,
                              linestyles=ls,
                              )
        if pnum == None:
            self.patches.append(cnt)
            self.raise_pcount()
            self.pnum.set(val + 1)
        else:
            self.patches[pnum] = cnt
        self.gcanvas.draw()

    def remove_patch(self):
        """
        Removes last patch from computing and from axes.
        """
        val = self.pcount - 1
        if val > 0:
            patch = self.patches.pop()
            self.clear_patch(patch)
            del (patch)
            self.decrease_pcount()
            self.plot(self.scale.get())
            pnum = int(self.pnum.get())
            if pnum > val:
                self.pnum.set(val)

    def move_limits(self, *args):
        pnum = self.get_pnum()
        p = self.patches[pnum]
        min_ = self.minsbox.get()
        max_ = self.maxsbox.get()
        self.clear_patch(p)
        self.make_patch(min_, max_, pnum)

    def compute(self, *args):
        style = self.style
        #        col = self.col
        self.ax.set_position([0.125, 0.125, 0.35227272727272724, 0.755])
        self.ax2.cla()
        self.ax2.set_axis_on()
        self.ax2.set_xlabel('Time [s]')
        self.ax2.set_ylabel('Pixel normalized power [-]')
        self.ax2.yaxis.tick_right()
        self.ax2.yaxis.set_label_position('right')
        count = self.pcount
        lines = {}
        for i in self.diags:
            lines[i] = []
            self.ax2.plot([], c='k', ls=style[i], label=i)

        for i in range(count):
            min_, max_ = self.get_levels(i)
            for d in self.diags:
                line = []
                for val in range(len(self.tvecs[d])):
                    mval = self.get_mval(val)
                    ge = self.mdata[mval] > min_
                    le = self.mdata[mval] <= max_
                    inds = le * ge
                    try:
                        line.append(np.sum(self.data[d][val, inds]))
                    except IndexError as e:
                        print(e)
                        self.inds = inds
                        print(inds)
                lines[d].append(line)
                self.ax2.plot(self.tvecs[d], lines[d][i], c='C{}'.format(i), ls=style[d])
        self.ax.legend()
        self.ax2.legend()
        ind = self.scale.get()
        self.vline = self.ax2.axvline(self.vals[ind], c='k',
                                      ls='dashed', lw=1)
        y = self.ax2.get_ylim()[1] * 1.01
        x = self.vals[ind]
        self.text = self.ax2.text(x, y, str(x))
        self.gcanvas.draw()

    def save(self, fname=None):
        if fname == None:
            fname = 'Graphs/'
        if not os.path.exists(fname):
            os.makedirs(fname)
        v = self.scale.get()
        self.fig.savefig(fname +
                         '{} at {:.0f} {:.0f}-{:.0f}ms contours.png'.format(
                         # '{} at {:.0f} {:.0f}-{:.0f}ms contours.eps'.format(
                             self.shot, 1000 * self.vals[v], 1000 * self.vals[0], 1000 * self.vals[-1]))

    def update_data(self, ndata, **kw):
        GraphRgb.update_data(self, ndata, **kw)
        if len(self.fig.axes) > 1:
            self.ax.set_position(self.ax_orig)
            self.ax2.cla()
            self.ax2.set_axis_off()
        self.ax.lines.clear()
        for i in self.diags:
            self.ax.plot([], c=self.col[i], ls='', marker='s', label=i)
        self.ax.legend()
        self.gcanvas.draw()

    def plot(self, val, *args):
        inds = self.get_rgbval(val)
        b = self.data['bol'][inds[0]]
        n = self.data['ntr'][inds[1]]
        s = self.data['sxr'][inds[2]]
        rgb = np.stack((b, n, s), axis=2)
        rgb = rgb / np.max(rgb)
        cmy = np.ones_like(rgb) - rgb
        rgb = cmy
        if self.grids[0]:
            self.ax.set_xlabel('R [m]')
        else:
            self.ax.set_xlabel('Pixel')
        if self.grids[0]:
            self.ax.set_ylabel('z [m]')
        else:
            self.ax.set_ylabel('Pixel')

        if len(self.ax.images) == 0:
            self.ax.imshow(rgb, origin='lower',
                           #            self.ax.imshow(cmy, origin='lower',
                           extent=self.extent,
                           )
        else:
            self.ax.images[0].set_data(rgb)
        #            self.ax.images[0].set_data(cmy)
        #            self.ax.images[0].set_extent([self.xmin, self.xmax, self.ymin, self.ymax])
        if hasattr(self, 'magcont'):
            for i in self.magcont.collections.copy():
                self.ax.collections.remove(i)
        if self.magvar.get():
            mval = self.get_mval(val)
            self.magcont = self.ax.contour(self.mdata[mval], [1.0],
                                           extent=self.extent,
                                           # colors='w',
                                           colors='k',
                                           linewidths=1)
        if len(self.ax2.texts) > 0:
            x = self.vals[val]
            self.vline.set_xdata(x)
            self.text.set_x(x)
            self.text.set_text(str(x))
        #        print(self.ax.collections)
        for i, p in enumerate(self.patches):
            min_, max_ = self.get_levels(i)
            #            print(p)
            self.clear_patch(p)
            self.make_patch(min_, max_, i)
        self.gcanvas.draw()

    def save_nth(self):
        """
        Saves every nth slice, where n is defined by snvar tk.Stringvar
        asociated with snsbox tk.Spinbox widget.
        """
        i = 0
        n = int(self.snvar.get())
        while i < len(self.vals):
            self.go_to(i)
            self.plot(i)
            self.save(fname='Graphs/cntseq/{}/'.format(self.shot))
            i += n
