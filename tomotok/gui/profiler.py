# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Averaging of emissivity in bins shaped according to magnetic flux reconstruction
"""
import os
import tkinter as tk
from tkinter import ttk

import numpy as np

from tomotok.core.phantoms import iso_psi
from .widgets_tomo import GraphRgb


class GraphProfiler(GraphRgb):
    """
    Based on GraphRgb class. Usefull for observing power profile in respect to
    magentic flux surfaces.
    
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
        if not 'title' in kw:
            self.title('Profiler')
        if not hasattr(self, 'mdata'):
            print('Found no magnetics data, using isotropic flux.')
            self.mdata = iso_psi(self.shape[2], self.shape[1])
            self.mdata = self.mdata.reshape(1, self.shape[1], self.shape[2])
            self.mtvec = np.zeros(1)
        self.ax_orig = self.ax.get_position()
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_axis_off()
        dummy_ax = self.fig.add_subplot(121)
        self.ax_left = dummy_ax.get_position()
        dummy_ax.remove()

        self.pprop = ttk.Frame(self, padding=(3, 3, 3, 3))  # , relief = 'solid')
        xrow = 1
        self.pprop.grid(row=0, column=2, rowspan=2, sticky='NSWE')
        self.stavar = tk.StringVar(value=0)
        self.stasbox = tk.Spinbox(self.pprop, from_=0, to=1,
                                  increment=.1, textvariable=self.stavar,
                                  command=self.plot_contour, width=5)
        self.stasbox.grid(row=xrow, column=1)
        self.stalbl = ttk.Label(self.pprop, text='start')
        self.stalbl.grid(row=xrow, column=0)

        self.stovar = tk.StringVar(value=1)
        self.stosbox = tk.Spinbox(self.pprop, from_=0, to=1.5,
                                  increment=.1, textvariable=self.stovar,
                                  command=self.plot_contour, width=5)
        self.stosbox.grid(row=xrow + 1, column=1)
        self.stolbl = ttk.Label(self.pprop, text='stop')
        self.stolbl.grid(row=xrow + 1, column=0)

        self.numvar = tk.StringVar(value=10)
        self.numsbox = tk.Spinbox(self.pprop, from_=1, to=50,
                                  increment=1, textvariable=self.numvar,
                                  command=self.plot_contour, width=5)
        self.numsbox.grid(row=xrow + 2, column=1)
        self.numlbl = ttk.Label(self.pprop, text='#')
        self.numlbl.grid(row=xrow + 2, column=0)
        self.cmpbtn = ttk.Button(self.pprop, text='Compute',
                                 command=self.compute)
        self.cmpbtn.grid(row=xrow + 3, column=0, columnspan=2, sticky='WE')
        #        self.levels = self.get_levels()

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

        self.expanded = False
        self.steps_plt = {}
        self.style = {'bol': 'solid', 'ntr': 'dashdot', 'sxr': 'dashed'}

    def get_levels(self, *args):
        """
        Gets levels from widgets for plotting contours
        """
        sta = np.float(self.stasbox.get())
        sto = np.float(self.stosbox.get())
        num = np.int(self.numsbox.get())
        return np.linspace(sta, sto, num)

    def plot_contour(self, *args):
        """
        Plots contours to highlight flux locations separating profiled regions.
        """
        self.ax.collections.clear()
        self.levels = self.get_levels()
        val = self.scale.get()
        mval = self.get_mval(val)
        self.cont = self.ax.contour(self.mdata[mval],
                                    levels=self.levels,
                                    extent=self.extent,
                                    # colors='w',
                                    colors='k',
                                    linewidths=1,
                                    )
        if self.magvar.get():
            mval = self.get_mval(val)
            self.mcont = self.ax.contour(self.mdata[mval], [1.0],
                                         extent=self.extent,
                                         # colors='w',
                                         colors='k',
                                         linewidths=1,
                                         linestyles='dashed')
        self.gcanvas.draw()

    #    def compute_cmd(self, *args):
    #        self.compute(*args)
    #        self.gcanvas.draw()

    def compute(self, *args):
        """
        Computes radiated power in given regions.
        """
        self.ax.set_position(self.ax_left)
        self.ax2.set_axis_on()
        self.ax2.set_xlabel('Psi normalized [-]')
        #        self.ax2.set_ylabel('Pixel normalized power [-]')
        self.ax2.set_ylabel('Normalized power density [-]')
        self.ax2.yaxis.tick_right()
        self.ax2.yaxis.set_label_position('right')

        self.res = []
        tmx = 0
        for i, val in enumerate(self.vals):
            mx = 0
            res = {}
            mval = self.get_mval(i)
            for di in self.diags:
                res[di] = []
            rgbind = list(self.get_rgbval(i))
            tind = dict(zip(['bol', 'ntr', 'sxr'], rgbind))
            for l in range(len(self.levels) - 1):
                ge = self.mdata[mval] >= self.levels[l]
                le = self.mdata[mval] < self.levels[l + 1]
                inds = le * ge
                for d in self.diags:
                    #                res[d].append(np.sum(self.data[d][inds,tind[d]]))
                    res[d].append(np.average(self.data[d][tind[d], inds]))
                    mx = np.max([mx, np.max(res[d])])
            self.res.append(res)
            tmx = max(tmx, mx)
        self.centers = np.diff(self.levels)[0] / 2 + self.levels[:-1]

        self.ax.legend()

        self.y2max = tmx
        self.ax2.set_ylim(0, 1.05 * self.y2max)
        self.ax2.set_xlim(float(self.stasbox.get()), float(self.stosbox.get()))
        #        if not self.expanded:
        #            self.gcanvas.draw()
        self.expanded = True
        self.plot(self.scale.get())
        self.ax2.legend()

    def save(self, fname='Graphs/'):
        if not os.path.exists(fname):
            os.makedirs(fname)
        v = self.scale.get()
        sta = self.stasbox.get()
        sto = self.stosbox.get()
        num = self.numsbox.get()
        self.fig.savefig(fname +
                         '{} at {:.0f}ms {}-{}-{}psi profiler.png'.format(
                         # '{} at {:.0f}ms {}-{}-{}psi profiler.eps'.format(
                             self.shot, 1000 * self.vals[v], sta, sto, num))

    def update_data(self, ndata, **kw):
        GraphRgb.update_data(self, ndata, **kw)
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
            #            self.ax.imshow(cmy, origin = 'lower',
            self.ax.imshow(rgb, origin='lower',
                           extent=self.extent
                           )
        #            shp = np.shape(rgb)
        #            self.ax.set_xlim(0,shp[1])
        #            self.ax.set_ylim(0,shp[0])
        else:
            self.ax.images[0].set_data(rgb)
            #            self.ax.images[0].set_data(cmy)
            self.ax.images[0].set_extent(self.extent)
        self.plot_contour()
        if self.expanded:
            self.fig.suptitle('#{} Profile @ {:.0f}ms'.format(
                self.shot, 1000 * self.vals[self.scale.get()]))
            self.ax2.lines.clear()
            #            self.ax2.clear()
            #            self.ax2.set_axis_on()
            #            self.ax2.set_xlabel('Psi normalized [-]')
            ##            self.ax2.set_ylabel('Pixel normalized power [-]')
            #            self.ax2.set_ylabel('Normalized power denplot_contoursity [-]')
            #            self.ax2.yaxis.tick_right()
            #            self.ax2.yaxis.set_label_position('right')
            for d in self.diags:
                #                if len(self.ax2.lines) > 0:
                #                    self.ax2.lines.remove(self.steps_plt[d])
                #                self.ax2.plot(self.centers, res[d], c=self.col[d],
                #                          ls=self.style[d], label=d)
                self.steps_plt[d] = self.ax2.step(self.centers,
                                                  self.res[val][d], c=self.col[d],
                                                  where='mid',
                                                  #                                      linestyle=self.style[d],
                                                  label=d)
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
            self.save(fname='Graphs/prfseq/{}/'.format(self.shot))
            i += n
