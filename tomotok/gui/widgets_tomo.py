# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Module with tkinter widgets for basic tomography analysis.
"""
import os
import tkinter as tk
from tkinter import ttk

import matplotlib.colorbar as cbar
import numpy as np

from .widgets import GraphWindow


# TODO REC normalize data?


class DiagGraphWindow(GraphWindow):
    """
    Based on GraphWindow class from widgets.py
    Used for displaying results of single diagnostic tomography

    See documentation of GraphWindow for information about other keyword arguments.
    
    Parameters
    ----------
    parent : TKInter widget
        Default is None. Used in GUI interface to specify hierarchy.
    data : dict
        Contains data, default is empty dict.
            
    **kw
        title : str {'Bolometry', 'Neutrons', 'SXR'}
            Name of graph window. If dname is not specified, title is used to 
            determin what key from provided data to load.
        dname : str {'bol', 'ntr', 'sxr'}
            Shorter string to specify diagnostic for graph window. Though not recommended, both title and dname can be used.
            If so, dname and title should always match to avoid inconsistency of title and data.
    
    See Also
    --------
    GraphWindow
    """

    def __init__(self, parent=None, data=None, **kw):
        GraphWindow.__init__(self, parent, **kw)
        if data is None:
            data = {}
        self.keys = kw.keys()
        if 'title' in self.keys:
            title = kw['title']
        else:
            title = 'Unknown diagnostic'
        if 'dname' in self.keys:
            dname = kw['dname']
        else:
            if title == 'Bolometry':
                dname = 'bol'
            elif title == 'Neutrons':
                dname = 'ntr'
            elif title == 'SXR':
                dname = 'sxr'
            else:
                dname = ''
                print('Unknown diagnostic name. Can not read data dict.',
                      'Check title or dname keyword arguments.')
        if 'shot' in self.keys:
            self.shot = kw['shot']
        elif 'shot' in data:
            self.shot = data['shot']
        else:
            self.shot = 'noshot'
        self.magvar = tk.BooleanVar(value=0)
        self.magchb = ttk.Checkbutton(self.bbox, text='Magnetics',
                                      variable=self.magvar, onvalue=True,
                                      command=self.update_graph)
        self.magchb.grid(row=self.bb_row, sticky='W')
        self.bb_row_raise()
        self.bndvar = tk.BooleanVar(value=1)
        self.bndchb = ttk.Checkbutton(self.bbox, text='Borders',
                                      variable=self.bndvar, onvalue=True,
                                      command=self.update_graph)
        self.bndchb.grid(row=self.bb_row, sticky='W')
        self.bb_row_raise()
        self.nrmvar = tk.BooleanVar(value=1)
        self.nrmchb = ttk.Checkbutton(self.bbox, text='Normalize',
                                      variable=self.nrmvar, onvalue=True,
                                      command=self.update_graph)
        self.nrmchb.grid(row=self.bb_row, sticky='W')
        self.bb_row_raise()
        self.title(title)
        self.ptitle = title
        self.dname = dname
        self.update_data(data, **kw)
        self.cax = cbar.make_axes(self.ax)[0]

    def plot(self, val, **kw):
        self.ax.cla()
        self.cax.cla()
        self.ax.set_xlabel('R [m]')
        self.ax.set_ylabel('z [m]')
        self.ax.set_title('{} at {:.4f} s'.format(self.ptitle, self.vals[val]))
        if self.nrmvar.get():
            im = self.ax.imshow(self.data[val], origin='lower', extent=[self.xmin, self.xmax, self.ymin, self.ymax],
                                vmin=self.vmin, vmax=self.vmax)
        else:
            im = self.ax.imshow(self.data[val], origin='lower', extent=[self.xmin, self.xmax, self.ymin, self.ymax])
        if self.magvar.get():
            mval = self.get_mval(val)
            self.ax.contour(self.mdata[mval], [1.0],
                            colors='w',
                            extent=[self.xmin, self.xmax, self.ymin, self.ymax],
                            linewidths=1,
                            )
        if self.bndvar.get():
            self.ax.plot(self.bnd[:, 0], self.bnd[:, 1], linewidth=2.0, color='k')
        self.fig.colorbar(im, cax=self.cax, ax=self.ax)
        self.gcanvas.draw()

    def update_data(self, ndata, **kw):
        """
        Loads new data.
        
        Parameters
        ----------
        ndata : dict
            New data to be loaded to Graph window.
        """
        if 'tind' not in ndata:
            ndata['tind'] = {}
        if self.dname in ndata['tind']:
            tind = ndata['tind'][self.dname]
        else:
            tind = range(len(ndata[self.dname]['tvec']))
        self.data = ndata[self.dname]['Recs'][tind, ...]
        keys = ndata[self.dname].keys()
        shape = np.shape(self.data)
        if 'R' in keys:
            x = ndata[self.dname]['R']
            dx = (x[1] - x[0]) / 2
            self.xmin = x[0] - dx
            self.xmax = x[-1] + dx
        else:
            self.xmin = 0
            self.xmax = shape[1]
        if 'Z' in keys:
            z = ndata[self.dname]['Z']
            dz = (z[1] - z[0]) / 2
            self.ymin = z[0] - dz
            self.ymax = z[-1] + dz
        else:
            self.ymin = 0
            self.ymax = shape[0]
        if 'mag' in ndata:
            self.mdata = ndata['mag']['psiRz'][ndata['mtind']]
            self.mtvec = ndata['mag']['tflux'][ndata['mtind']]
            self.magchb.state(['!disabled'])
            self.magvar.set(1)
        else:
            self.magvar.set(0)
            self.magchb.state(['disabled'])
        try:
            self.bnd = np.loadtxt('tomotok/output/graphs/border.txt')
        except IOError:
            self.bndvar.set(0)
            self.bndchb.state(['disabled'])
        self.vmin = np.min(self.data)
        self.vmax = np.max(self.data)
        self.set_vals(vals=ndata[self.dname]['tvec'][tind])

    def get_mval(self, val):
        """
        Returns index for ploting magnetic data from index of reconstrution.
        Should not be called if magnetics data are not loaded.
        
        Parameters
        ----------
        val : int
            Index of reconstrition.
        """
        try:
            out = np.argmin(np.abs(self.mtvec - self.vals[val]))
        except:
            print('get_mval failure')
            out = 0
        return out

    def save(self, *args):
        fname = 'Graphs'
        if not os.path.exists(fname):
            os.makedirs(fname)
        val = self.scale.get()
        tm = np.round(1000 * self.vals[val])
        self.fig.savefig(os.path.join(fname, '{}at{:.0f}ms_{}.png'.format(
            self.shot, tm, self.dname)))


class GraphRgb(GraphWindow):
    """
    Based on GraphWindow class from widgets.py. Usefull for comparing
    tomography results from different diagnostics simultaneously.

    See documentation of Graf_window for information about other keyword arguments.
    
    Parameters
    ----------
    parent : TKInter class parent
        Default is None.
    data : dict
        Contains data, default is empty dict.
    
    Keywords
    --------
    title : str
        Name of graph window. Is used for saving. Default is 'RGB plot'
    """

    def __init__(self, parent=None, data=None, **kw):
        GraphWindow.__init__(self, parent, **kw)
        if data is None:
            data = {}
        # self.col = {'bol': 'r', 'ntr': 'g', 'sxr': 'b'}
        self.col = {'bol': 'c', 'ntr': 'm', 'sxr': 'y'}
        self.magvar = tk.BooleanVar(value=1)
        self.magchb = ttk.Checkbutton(self.bbox, text='Magnetics',
                                      variable=self.magvar, onvalue=True,
                                      command=self.update_graph)
        self.magchb.grid(row=self.bb_row, sticky='W')
        self.bb_row_raise()
        if 'title' in kw:
            self.title(kw['title'])
        else:
            self.title('RGB plot')
        #            self.title('CMY plot')
        self.update_data(data, vals=self.vals)

    def update_data(self, ndata, **kw):
        """
        Loads new data.
        
        Parameters
        ----------
        ndata : dict
            New data to be loaded to Graph window.
        """
        #        self.data = {}
        self.tvecs = {}
        self.diags = []
        try:
            self.shot = ndata['shot']
        except KeyError:
            self.shot = 0
        keys = ndata.keys()
        if 'tind' not in keys:
            ndata['tind'] = {}
        if 'bol' in keys:
            if 'bol' in ndata['tind']:
                tind = ndata['tind']['bol']
            else:
                tind = range(len(ndata['bol']['tvec']))
            bol = ndata['bol']['Recs'][tind, ...]
            nbol = bol / np.max(bol)
            nbol = np.where(nbol < 0, 0, nbol)
            self.tvecs['bol'] = ndata['bol']['tvec'][tind]
            self.diags.append('bol')
        else:
            bol = None
            nbol = None
            self.tvecs['bol'] = np.zeros(1)
        if 'ntr' in keys:
            if 'ntr' in ndata['tind']:
                tind = ndata['tind']['ntr']
            else:
                tind = range(len(ndata['sxr']['tvec']))
            ntr = ndata['ntr']['Recs'][tind, ...]
            nntr = ntr / np.max(ntr)
            nntr = np.where(nntr < 0, 0, nntr)
            self.tvecs['ntr'] = ndata['ntr']['tvec'][tind]
            self.diags.append('ntr')
        else:
            ntr = None
            nntr = None
            self.tvecs['ntr'] = np.zeros(1)
        if 'sxr' in keys:
            if 'sxr' in ndata['tind']:
                tind = ndata['tind']['sxr']
            else:
                tind = range(len(ndata['sxr']['tvec']))
            sxr = ndata['sxr']['Recs'][tind, ...]
            nsxr = sxr / np.max(sxr)
            nsxr = np.where(nsxr < 0, 0, nsxr)
            self.tvecs['sxr'] = ndata['sxr']['tvec'][tind]
            self.diags.append('sxr')
        else:
            sxr = None
            nsxr = None
            self.tvecs['sxr'] = np.zeros(1)
        if 'mag' in ndata:
            self.mdata = ndata['mag']['psiRz'][ndata['mtind']]
            self.mtvec = ndata['mag']['tflux'][ndata['mtind']]
            self.magvar.set(True)
            self.magchb.state(['!disabled'])
        else:
            self.magvar.set(False)
            self.magchb.state(['disabled'])
        if 'mag' in kw:
            if not kw['mag']:
                self.magvar.set(0)
                self.magchb.state(['disabled'])
        self.data = {'bol': nbol, 'ntr': nntr, 'sxr': nsxr}
        self.odata = {'bol': bol, 'ntr': ntr, 'sxr': sxr}
        self.key = self.diags[0]
        tvec = np.array(self.tvecs[self.key])
        keys = ndata[self.key]
        shape = np.shape(self.data[self.key])
        self.grids = [False, False]
        if 'R' in keys:
            x = ndata[self.key]['R']
            dx = (x[1] - x[0]) / 2
            self.xmin = x[0] - dx
            self.xmax = x[-1] + dx
            self.dx = 2 * dx
            self.grids[0] = True
        else:
            self.xmin = 0
            self.xmax = shape[1]
            self.dx = 1
        if 'Z' in keys:
            z = ndata[self.key]['Z']
            dz = (z[1] - z[0]) / 2
            self.ymin = z[0] - dz
            self.ymax = z[-1] + dz
            self.dy = 2 * dz
            self.grids[1] = True
        else:
            self.ymin = 0
            self.ymax = shape[0]
            self.dy = 1
        self.extent = [self.xmin, self.xmax, self.ymin, self.ymax]
        self.shape = shape
        if len(self.diags) < 3:
            shape = np.array(shape)
            shape[0] = 1
            if not 'bol' in self.diags:
                nbol = np.zeros(shape)
                bol = np.zeros(shape)
                self.tvecs['bol'] = np.zeros(1)
            if not 'ntr' in self.diags:
                nntr = np.zeros(shape)
                ntr = np.zeros(shape)
                self.tvecs['ntr'] = np.zeros(1)
            if not 'sxr' in self.diags:
                nsxr = np.zeros(shape)
                sxr = np.zeros(shape)
                self.tvecs['sxr'] = np.zeros(1)
            self.data = {'bol': nbol, 'ntr': nntr, 'sxr': nsxr}
            self.odata = {'bol': bol, 'ntr': ntr, 'sxr': sxr}
        self.set_vals(vals=tvec)

    #        self.plot(0)

    def get_mval(self, val):
        """
        Returns index for ploting magnetic data from index of reconstrution.
        Should not be called if magnetics data are not loaded.
        
        Parameters
        ----------
        val : int
            Index of reconstrition.
        """
        try:
            out = np.argmin(np.abs(self.mtvec - self.vals[val]))
        except:
            print('get_mval failure')
            out = 0
        return out

    def get_rgbval(self, val):
        """
        Gets indexes for given timeslices. Independent on provided diagnostics.
        
        Returns
        -------
        numpy.ndarray of ints
            Contains indexes of nearest tim vector values
            to reference diagnostic.
        """
        out = np.zeros(3, dtype=np.int)
        out[0] = np.argmin(np.abs(self.tvecs['bol'] - self.vals[val]))
        out[1] = np.argmin(np.abs(self.tvecs['ntr'] - self.vals[val]))
        out[2] = np.argmin(np.abs(self.tvecs['sxr'] - self.vals[val]))
        return out

    def plot(self, val, *args):
        inds = self.get_rgbval(val)
        b = self.data['bol'][inds[0]]
        n = self.data['ntr'][inds[1]]
        s = self.data['sxr'][inds[2]]
        rgb = np.stack((b, n, s), axis=2)
        cmy = np.ones_like(rgb) - rgb
        #        cmy = np.power(cmy, 2)
        self.ax.cla()
        for i in self.diags:
            self.ax.plot([], c=self.col[i], ls='', marker='s', label=i)
        self.ax.legend()
        self.ax.set_xlabel('R [m]')
        self.ax.set_ylabel('z [m]')
        self.ax.imshow(rgb, extent=[self.xmin, self.xmax, self.ymin, self.ymax], origin='lower')
        self.ax.imshow(cmy, extent=[self.xmin, self.xmax, self.ymin, self.ymax], origin='lower')
        if self.magvar.get():
            mval = self.get_mval(val)
            self.ax.contour(self.mdata[mval], [1.0],
                            # colors='w',
                            colors='k',
                            linewidths=1,
                            extent=[self.xmin, self.xmax, self.ymin, self.ymax])
        self.gcanvas.draw()

    def save(self):
        fname = 'Graphs'
        val = self.scale.get()
        if not os.path.exists(fname):
            os.makedirs(fname)
        self.fig.savefig(os.path.join(fname, '{} RGB plot at {:.0f}ms.png'.format(self.shot, self.vals[val] * 1000)))


class GraphPower(GraphWindow):
    """
    Based on GraphWindow class from widgets.py
    Vals are not time steps for this class but types of diagnotics to plot 
    power emitted or to show correlations of these.
    
    Parameters
    ----------
    parent : TKInter class parent, optional
        Default is None.
    data : dict, optional
        Contains data, default is empty dict.
    """

    # TODO correlation shifts?
    def __init__(self, parent=None, data=None, **kw):
        # self.vals = ['all', 'bol', 'ntr', 'sxr', 'preproc' ,'corel', 'shifts']
        GraphWindow.__init__(self, parent, **kw)
        if data is None:
            data = []
        self.title('Power graph')
        self.normvar = tk.BooleanVar(value=0)
        self.normchb = ttk.Checkbutton(self.bbox, text='Normalize',
                                       variable=self.normvar, onvalue=True,
                                       command=self.normalize)
        self.normchb.grid(row=self.bb_row, column=0)
        self.bb_row_raise()
        self.update_data(data, )
        self.plot(0)

    def plot(self, val, **kw):
        """
        Prepares data for matplotlib and draws them onto canvas
        
        Parameters
        ----------
        val : int
            number of slice in this case meaning type of analysis
        
        """
        if self.normvar.get():
            nrm = ' normalized'
        else:
            nrm = ''
        # col = {'bol': 'r', 'ntr': 'g', 'sxr': 'b'}
        col = {'bol': 'c', 'ntr': 'm', 'sxr': 'y'}
        ndiag = len(self.diags)
        ncor = len(self.cdata)
        shot = self.shot
        if val == 0:
            self.ax.cla()
            self.ax.set_title('#{} Total power{}'.format(shot, nrm))
            for i in range(ndiag):
                diag = self.diags[i]
                self.ax.plot(self.tvecs[i], self.data[i],
                             label=diag, color=col[diag])
            self.ax.legend()
            self.ax.set_xlabel('Time [s]')
            self.ax.set_ylabel('Power [{}]'.format(np.where(
                len(nrm) > 0, '-', 'Watt')))
            self.gcanvas.draw()
        elif 0 < val < ndiag:
            diag = self.vals[val]
            self.ax.cla()
            self.ax.set_title('#{} Total power of {}{}'.format(shot, diag, nrm))
            self.ax.set_xlabel('Time [s]')
            self.ax.set_ylabel('Power [{}]'.format(np.where(
                len(nrm) > 0, '-', 'Watt')))
            self.ax.plot(self.tvecs[val - 1], self.data[val - 1],
                         color=col[diag])
            self.gcanvas.draw()
        elif val == ndiag:
            self.ax.cla()
            self.ax.set_title('Preprocessed signals')
            self.ax.set_xlabel('Time [s]')
            for i in range(ndiag):
                diag = self.diags[i]
                self.ax.plot(self.tvecs[i], self.tmpdata[i],
                             label='Preproc ' + diag,
                             color=col[diag])
            self.ax.legend()
            self.gcanvas.draw()
        elif val == ndiag + 1:
            self.ax.cla()
            self.ax.set_title('Correlations{}'.format(nrm))
            self.ax.set_xlabel('Time shift [s]')
            lbls = ['corr(bolo,ntr)', 'corr(sxr,bolo)', 'corr(ntr,sxr)']
            for i in range(ncor):
                self.ax.plot(self.ctvec, self.cdata[i],
                             label=lbls[i], color=col[i])
            self.ax.legend()
            self.gcanvas.draw()
        elif val == ndiag + 2:
            self.ax.cla()
            self.ax.set_title('Shifted signals{}'.format(nrm))
            self.ax.set_xlabel('Arbitrary Time [s]')
            for i in range(ndiag):
                self.ax.plot(self.stvec[i], self.data[i],
                             label='{} shifted {:0.2} s'.format(
                                 self.diags[i],
                                 abs(self.shifts[i])),
                             color=col[i])
            self.ax.legend()
            self.gcanvas.draw()
        return

    def update_data(self, ndata, **kwargs):
        """
        Updates class data
        
        Parameters
        ----------
        ndata : dict
            Dictionary containign new data under keys 'bol', 'ntr', 'sxr'.
        """
        data = ndata
        self.diags = []
        self.sdata = []
        self.cdata = []
        self.tvecs = []
        try:
            self.shot = data['shot']
        except KeyError:
            self.shot = 0
        keys = ndata.keys()
        if 'tind' not in keys:
            ndata['tind'] = {}
        tkeys = ndata['tind'].keys()
        if 'bol' in keys:
            if 'bol' in tkeys:
                tind = ndata['tind']['bol']
            else:
                tind = np.arange(len(ndata['bol']['tvec']))
            if 'TOPO' in data['bol']:
                ptot = 'TOPO'
            else:
                ptot = 'TOBU'
            b = data['bol'][ptot][tind]
            self.sdata.append(b)
            self.tvecs.append(data['bol']['tvec'][tind])
            self.diags.append('bol')
        if 'ntr' in keys:
            if 'ntr' in tkeys:
                tind = ndata['tind']['ntr']
            else:
                tind = np.arange(len(ndata['ntr']['tvec']))
            if 'TOPO' in data['ntr']:
                ptot = 'TOPO'
            else:
                ptot = 'TOBU'
            n = data['ntr'][ptot][tind]
            self.sdata.append(n)
            self.tvecs.append(data['ntr']['tvec'][tind])
            self.diags.append('ntr')
        if 'sxr' in keys:
            if 'sxr' in tkeys:
                tind = ndata['tind']['sxr']
            else:
                tind = np.arange(len(ndata['sxr']['tvec']))
            if 'TOPO' in data['sxr']:
                ptot = 'TOPO'
            else:
                ptot = 'TOBU'
            s = data['sxr'][ptot][tind]
            self.sdata.append(s)
            self.tvecs.append(data['sxr']['tvec'][tind])
            self.diags.append('sxr')
        self.normd = None
        self.data = self.sdata
        self.set_vals(vals=['all'] + self.diags)
        self.normalize()

    def normalize(self, *args, **kw):
        """
        Switches data to normalized or standard, depending on norvar state.
        Normalizes data and if no normalized data are available.
        """
        if self.normd is None:
            self.normd = [i / np.max(i) for i in self.data]
        if self.normvar.get():
            self.data = self.normd
        else:
            self.data = self.sdata
        self.update_graph()

    def save(self):
        fname = 'Graphs'
        if not os.path.exists(fname):
            os.makedirs(fname)
        self.fig.savefig(os.path.join(fname, '{} {}.png'.format(self.shot, self.ax.get_title())))
        # self.fig.savefig(os.path.join(fname, '{} {}.eps'.format(self.shot, self.ax.get_title())))
