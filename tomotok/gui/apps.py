# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Module with tkinter apps for computing tomography reconstructions.
"""
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter import filedialog

import h5py
import numpy as np

from .widgets import MasterControls
from .widgets_tomo import DiagGraphWindow, GraphPower, GraphRgb
from .roi import GraphRectangles, GraphContours
from .profiler import GraphProfiler
from .merfer import GraphMerfer
from .menus import AnalyzerMenu, ExpandMenu, SelectorMenu


class AnalyzerApp(ttk.Frame):
    def __init__(self, master=None):
        master.rowconfigure(0, weight=1)
        master.columnconfigure(0, weight=1)
        master.title('Results analyzer')
        master.bind('<Return>', self.do)
        ttk.Frame.__init__(self, master, padding=(3, 3, 3, 3))
        self.grid()
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.menu = AnalyzerMenu(self)
        self.menu.grid(row=0, column=0, columnspan=2, sticky='NSWE')
        self.but = ttk.Button(self, text='do', command=self.do)
        self.but.grid(row=2, column=0, sticky='W')
        self.graf_bol = tk.Frame()
        self.graf_bol.destroy()
        self.graf_ntr = tk.Frame()
        self.graf_ntr.destroy()
        self.graf_sxr = tk.Frame()
        self.graf_sxr.destroy()
        self.graf_rgb = tk.Frame()
        self.graf_rgb.destroy()
        self.graf_rec = tk.Frame()
        self.graf_rec.destroy()
        self.graf_prf = tk.Frame()
        self.graf_prf.destroy()
        self.graf_pwr = tk.Frame()
        self.graf_pwr.destroy()
        self.graf_cnt = tk.Frame()
        self.graf_cnt.destroy()
        self.graf_mrf = tk.Frame()
        self.graf_mrf.destroy()
        self.mc = tk.Frame()
        self.mc.destroy()
        self.exp = None
        self.data = {}
        self.shot = 0
        self.resolution = None
        self.form = None

    def do(self, *args):
        """
        Loads data from files and creates requested graphs and analytical widgets.
        """
        # TODO hdf loading or selection of file and suffix detection
        stime = float(self.menu.stimevar.get())
        etime = float(self.menu.etimevar.get())
        self.data['tind'] = {}
        self.data['shot'] = self.menu.shotvar.get()
        self.shot = self.data['shot']
        self.resolution = self.menu.rescbox.get()
        names = {'bol': 'bolometers',
                    'ntr': 'neutrons',
                    'sxr': 'sxr',
                    }
        switches = {
            'bol': self.menu.bolvar,
            'ntr': self.menu.ntrvar,
            'sxr': self.menu.sxrvar,
        }
        tokname = self.menu.tokvar.get().capitalize()
        path = Path('Tomodata/{}/Output'.format(tokname))
        if tokname == 'Jet':
            for d in names:
                name = names[d]
                if switches[d].get():
                    try:
                        name = '{}/tomoresults_{}_{}'.format(name.capitalize(), self.shot, name)
                        floc = str(path / name)
                        res = self.resolution.split('x')
                        self.data[d] = {}
                        with h5py.File(floc, 'r') as fset:
                            tvec = fset['time'][...]
                            self.data[d]['tvec'] = tvec
                            self.data[d]['R'] = fset['r'][...]
                            self.data[d]['Z'] = fset['z'][...]
                            resh = [len(tvec), int(res[0]), int(res[1])]
                            tmp = fset['recs'][...]
                            tmp = tmp.reshape(resh)
                            self.data[d]['Recs'] = tmp
                            tmp = fset['ptot'][...]
                            tmp = tmp.values.reshape(len(tvec))
                            self.data[d]['TOPO'] = tmp
                            tmp = fset['used_data'][...]
                            tmp = tmp.values.reshape(-1, len(tvec))
                            self.data[d]['meli'] = tmp
                            tmp = fset['retrofit'][...]
                            tmp = tmp.values.reshape(-1, len(tvec))
                            self.data[d]['bcli'] = tmp
                            self.data['tind'][d] = (stime <= self.data[d]['tvec']) * (self.data[d]['tvec'] <= etime)
                    except Exception as e:
                        print('Unable to load {} data.'.format(names[d]))
                        print(e)
        elif tokname == 'Compass':
            name = 'axuv'
            d = 'bol'
            name = '{}/tomoresults_{}_{}'.format(name.capitalize(), self.shot, name)
            try:
                floc = str(path / name)
                res = self.resolution.split('x')
                self.data[d] = dict()
                with h5py.File(floc, 'r') as fset:
                    tvec = fset['time'][...]
                    self.data[d]['tvec'] = tvec
                    self.data[d]['R'] = fset['r'][...]
                    self.data[d]['Z'] = fset['z'][...]
                    resh = [len(tvec), int(res[0]), int(res[1])]
                    tmp = fset['recs'][...]
                    tmp = tmp.reshape(resh)
                    self.data[d]['Recs'] = tmp
                    tmp = fset['ptot'][...]
                    tmp = tmp.values.reshape(len(tvec))
                    self.data[d]['TOPO'] = tmp
                    tmp = fset['used_data'][...]
                    tmp = tmp.values.reshape(-1, len(tvec))
                    self.data[d]['meli'] = tmp
                    tmp = fset['retrofit'][...]
                    tmp = tmp.values.reshape(-1, len(tvec))
                    self.data[d]['bcli'] = tmp
                    self.data['tind'][d] = (stime <= self.data[d]['tvec']) * (self.data[d]['tvec'] <= etime)
            except Exception as e:
                print('Unable to load {} data.'.format(name))
                print(e)
        else:
            print('Unrecognized tokamak name {}'.format(tokname))
        mpath = Path('Tomodata/Jet/Magnetics/')
        try:
            # TODO lims selector
            lims = '1.8-4.0--1.9-2.1'
            mname = 'Magnetic_Field_{shot}_{res}-{lims}.npz'.format(
                    shot=self.shot, res=self.resolution, lims=lims)
            self.data['mag'] = np.load(mpath / mname)
            mtvec = self.data['mag']['tflux']
            self.data['mtind'] = np.where((stime <= mtvec) & (mtvec <= etime))[0]
        except IOError as e:
            print('Could not load magnetics data: {}'.format(e))
        if self.menu.bolvar.get() and self.graf_bol.winfo_exists():
            self.graf_bol.update_data(self.data)
        else:
            self.graf_bol.destroy()
        if self.menu.ntrvar.get() and self.graf_ntr.winfo_exists():
            self.graf_ntr.update_data(self.data)
        else:
            self.graf_ntr.destroy()
        if self.menu.sxrvar.get() and self.graf_sxr.winfo_exists():
            self.graf_sxr.update_data(self.data)
        else:
            self.graf_sxr.destroy()
        if self.menu.rgbvar.get() or self.graf_rgb.winfo_exists():
            self.make_rgb()
        if self.menu.recvar.get() or self.graf_rec.winfo_exists():
            self.make_rec()
        if self.menu.prfvar.get() or self.graf_prf.winfo_exists():
            self.make_prf()
        if self.menu.pwrvar.get() or self.graf_pwr.winfo_exists():
            self.make_pwr()
        if self.menu.cntvar.get() or self.graf_cnt.winfo_exists():
            self.make_cnt()
        self.expand()

    def make_bol(self):
        """
        Makes Bolometry graph window or updates data in existing window.
        """
        if not self.graf_bol.winfo_exists():
            self.graf_bol = DiagGraphWindow(self, data=self.data, title='Bolometry')
        else:
            self.graf_bol.update_data(self.data)
        self.graf_bol.plot(0)

    def make_ntr(self):
        """
        Makes neutron graph window or updates data in existing window.
        """
        tind = self.data['tind']['ntr']
        tvec = self.data['ntr']['tvec'][tind]
        if not self.graf_ntr.winfo_exists():
            self.graf_ntr = DiagGraphWindow(self, data=self.data, vals=tvec, title='Neutrons')
        else:
            self.graf_ntr.update_data(self.data, vals=tvec)
        self.graf_ntr.plot(0)

    def make_sxr(self):
        """
        Makes Soft X-Ray graph window or updates data in existing window.
        """
        tind = self.data['tind']['sxr']
        tvec = self.data['sxr']['tvec'][tind]
        if not self.graf_sxr.winfo_exists():
            self.graf_sxr = DiagGraphWindow(self, data=self.data,
                                            vals=tvec, title='SXR')
        else:
            self.graf_sxr.update_data(self.data, vals=tvec)
        self.graf_sxr.plot(0)

    def make_rgb(self):
        """
        Makes RGB graph window or updates data in existing RGB graph window.
        """
        if not self.graf_rgb.winfo_exists():
            self.graf_rgb = GraphRgb(self, data=self.data)
        else:
            self.graf_rgb.update_data(self.data)
        self.graf_rgb.plot(0)

    def make_rec(self):
        """
        Makes rectangles graph window or updates data in existing window.
        """
        if not self.graf_rec.winfo_exists():
            self.graf_rec = GraphRectangles(self, data=self.data)
        else:
            self.graf_rec.update_data(self.data)
        self.graf_rec.plot(0)

    def make_prf(self):
        """
        Makes profiler graph window or updates data in existing window.
        """
        if not self.graf_prf.winfo_exists():
            self.graf_prf = GraphProfiler(self, data=self.data)
        else:
            self.graf_prf.update_data(self.data)
        self.graf_prf.plot(0)

    def make_pwr(self):
        """
        Makes power graph window or updates data in existing window.
        """
        if not self.graf_pwr.winfo_exists():
            self.graf_pwr = GraphPower(self, self.data)
        else:
            self.graf_pwr.update_data(self.data, )
    #        self.graf_pwr.plot(0)
    
    def make_mrf(self):
        """
        Makes Merfer graph window or updates data in existing RGB graph window.
        """
        if not self.graf_mrf.winfo_exists():
            self.graf_mrf = GraphMerfer(self, data=self.data)
        else:
            self.graf_mrf.update_data(self.data)
        self.graf_mrf.plot(0)

    def make_cnt(self):
        """
        Makes contours graph window or updates data in existing window.
        """
        if not self.graf_cnt.winfo_exists():
            self.graf_cnt = GraphContours(self, data=self.data)
        else:
            self.graf_cnt.update_data(self.data)

    def get_ctrl(self):
        """
        Checks for existing slave graphical windows and returns them in list

        Returns
        -------
        control : array
            Contains all slave graphs widgets.
        """
        control = []
        slaves = [self.graf_bol, self.graf_ntr, self.graf_sxr,
                  self.graf_rgb, self.graf_mrf]
        for i in slaves:
            if i.winfo_exists():
                control.append(i)
        return control

    def make_mc(self):
        """
        Makes Master controls widget for slave graphs.
        """
        control = self.get_ctrl()
        if len(control) > 0:
            tind = control[0].vals
            slices = len(tind)
            kw = {'vals': tind}
        else:
            kw = {}
            slices = 1
        if self.mc.winfo_exists():
            self.mc.ctrl = control
            self.mc.set_slices(slices, **kw)
        else:
            self.mc = MasterControls(self, slices, *control, **kw)

    def expand(self, **kw):
        """
        Executed after main 'Do' method.
        Makes buttons for recreating closed graph windows.
        """
        self.exp = ExpandMenu(self, padding=(3, 3, 12, 12), relief='solid')
        self.exp.grid(row=1, column=0, columnspan=3, sticky='NSWE')


class SelectorApp(AnalyzerApp):
    """
    Manually select hdf files to be used for comparative analysis.
    """
    def __init__(self, master=None):
        AnalyzerApp.__init__(self, master)
        self.menu.destroy()
        self.menu = SelectorMenu(self)
        self.menu.grid(row=0, column=0, columnspan=2, sticky='NSWE')
        # TODO
        # self.menu.var3.set('Output/mhd/tomoresults_6071_sxr')
        # self.menu.varm.set('Output/mhd/mag_mhd')
        self.fnames = list()

    def select1(self):
        nm = filedialog.askopenfilename(parent=self)
        self.menu.var1.set(nm)

    def select2(self):
        nm = filedialog.askopenfilename(parent=self)
        self.menu.var2.set(nm)

    def select3(self):
        nm = filedialog.askopenfilename(parent=self)
        self.menu.var3.set(nm)

    def selectm(self):
        nm = filedialog.askopenfilename(parent=self)
        self.menu.varm.set(nm)

    def do(self, *args, **kwargs):
        self.data = {}
        self.fnames = list()
        strvars = (self.menu.var1, self.menu.var2, self.menu.var3)
        stime = float(self.menu.stimevar.get())
        etime = float(self.menu.etimevar.get())
        self.data['tind'] = {}
        self.data['errors'] = {}
        for var in strvars:
            self.fnames.append(var.get())
        # TODO making dict from hdf as method of Tomoset class
        # for i, floc in enumerate(self.fnames):
        names = ('bol', 'ntr', 'sxr')
        for a in range(3):
            floc = self.fnames[a]
            d = names[a]
            self.data[d] = dict()
            if floc:
                with h5py.File(floc, 'r') as fset:
                    tvec = fset['time'][...]
                    self.data[d]['tvec'] = tvec
                    self.data[d]['R'] = fset['r'][...]
                    self.data[d]['Z'] = fset['z'][...]
                    # TODO change recs to values
                    self.data[d]['Recs'] = fset['recs'][...]
                    tmp = fset['ptot'][...]
                    tmp = tmp.reshape(len(tvec))
                    self.data[d]['TOPO'] = tmp
                    tmp = fset['used_data'][...]
                    tmp = tmp.reshape(-1, len(tvec))
                    self.data[d]['meli'] = tmp
                    tmp = fset['retrofit'][...]
                    tmp = tmp.reshape(-1, len(tvec))
                    self.data[d]['bcli'] = tmp
                    try:
                        self.data['errors'][d] = fset['errors'][...]
                    except KeyError:
                        pass
                    # TODO load only relevant time interval
                    self.data['tind'][d] = (stime <= self.data[d]['tvec']) * (self.data[d]['tvec'] <= etime)
                # TODO data loading and widget creation
                # TODO fnames to Path and check availability
        mloc = self.menu.varm.get()
        if mloc:
            magfield = np.load(mloc)
            self.data['mag'] = {}
            self.data['mag']['psiRz'] = magfield['values']
            self.data['mag']['tflux'] = magfield['time']
            mtvec = magfield['time']
            self.data['mtind'] = (stime <= mtvec) * (mtvec <= etime)

        self.exp = ttk.Frame(self)
        self.exp.grid(row=1, column=0, columnspan=3, sticky='NSWE')
        self.explbl = ttk.Label(self.exp, text='Create window:')
        self.explbl.grid(row=0, column=0, columnspan=2, sticky='W')
        #        for i in range(6):
        #            self.columnconfigure(i, weight = 1)
        bwid = 7
        self.bolbtn = ttk.Button(self.exp, text='bol', width=bwid, command=self.make_bol)
        self.bolbtn.grid(row=1, column=0)
        if not self.menu.var1.get():
            self.bolbtn.state(['disabled'])
        self.ntrbtn = ttk.Button(self.exp, text='ntr', width=bwid, command=self.make_ntr)
        self.ntrbtn.grid(row=1, column=1)
        if not self.menu.var2.get():
            self.ntrbtn.state(['disabled'])
        self.sxrbtn = ttk.Button(self.exp, text='sxr', width=bwid, command=self.make_sxr)
        self.sxrbtn.grid(row=1, column=2)
        if not self.menu.var3.get():
            self.sxrbtn.state(['disabled'])
        self.mrfbtn = ttk.Button(self.exp, text='mrf', width=bwid, command=self.make_mrf)
        self.mrfbtn.grid(row=1, column=4)
        self.rgbbtn = ttk.Button(self.exp, text='rgb', width=bwid, command=self.make_rgb)
        self.rgbbtn.grid(row=2, column=0)
        self.pwrbtn = ttk.Button(self.exp, text='pwr', width=bwid, command=self.make_pwr)
        self.pwrbtn.grid(row=2, column=1)
        self.recbtn = ttk.Button(self.exp, text='rec', width=bwid, command=self.make_rec)
        self.recbtn.grid(row=2, column=2)
        self.prfbtn = ttk.Button(self.exp, text='prf', width=bwid, command=self.make_prf)
        self.prfbtn.grid(row=2, column=3)
        self.cntbtn = ttk.Button(self.exp, text='cnt', width=bwid, command=self.make_cnt)
        self.cntbtn.grid(row=2, column=4)
        self.masterbtn = ttk.Button(self.exp, text='master', command=self.make_mc, width=bwid + 3)
        self.masterbtn.grid(row=3, column=0, columnspan=2, sticky='W')

    def make_mrf(self):
        if not self.graf_mrf.winfo_exists():
            self.graf_mrf = GraphMerfer(self, data=self.data, errors=self.data['errors'])
        else:
            self.graf_mrf.update_data(self.data, err=self.data['errors'])
        self.graf_mrf.plot(0)


def launcher(app='analyzer'):
    """
    Launches specified application and ensures stdout.

    Parameters
    ----------
    app : string
        Specifies which app should be launched. Legal options are
        and 'analyzer', 'a'. These are not case sensitive.
    """
    root = tk.Tk()
    upp = app.upper()
    if upp in ['A', 'ANALYZER']:
        app = AnalyzerApp(root)
    elif upp in ['S', 'SELECTOR']:
        app = SelectorApp(root)
    wdef = sys.stdout.write
    try:
        app.mainloop()
    finally:
        sys.stdout.write = wdef
