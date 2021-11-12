# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Contains frame widget subclasses prepared for use as selection menu in different GUI apps.
"""
import tkinter as tk
from tkinter import ttk


class AnalyzerMenu(ttk.Frame):
    """
    TTK Frame based class containing options for tomography.
    """

    def __init__(self, parent=None):
        ttk.Frame.__init__(self, parent)  # padding = (3,3,3,3))
        self.grid(sticky='WE')
        self.shot = ttk.Frame(self, padding=(3, 3, 12, 12), relief='solid')
        self.shot.grid(row=0, column=0, columnspan=2)
        self.tokvar = tk.StringVar()
        self.cboxlbl = ttk.Label(self.shot, text='Tokamak:')
        self.cboxlbl.grid(row=0, column=0, sticky='EW')
        toklist = ['JET', 'COMPASS']
        self.cbox = ttk.Combobox(self.shot, values=toklist, width=10,
                                 state='readonly',
                                 textvariable=self.tokvar)
        self.cbox.current(0)
        self.cbox['state'] = 'readonly'
        # self.cbox['state'] = 'disabled'
        self.cbox.grid(row=0, column=1, sticky='WE')
        shotrow = 0
        shotcol = 2
#        self.shotvar = tk.StringVar(value=92292)
#        self.shotvar = tk.StringVar(value=92400)
#        self.shotvar = tk.StringVar(value=91718)
        self.shotvar = tk.StringVar(value=94396)
        self.shotlbl = ttk.Label(self.shot, text='Shot:')
        self.shotent = ttk.Entry(self.shot, textvariable=self.shotvar,
                                 width=5)
        self.shotlbl.grid(row=shotrow, column=shotcol + 0, sticky='E')
        self.shotent.grid(row=shotrow, column=shotcol + 1, sticky='WE')
        self.reslbl = ttk.Label(self.shot, text='Resolution')
        self.reslbl.grid(row=1, column=0)
        self.rescbox = ttk.Combobox(self.shot, values=['55x100', '115x196', '110x200'], state='active')
        self.rescbox.grid(row=1, column=1)
        self.rescbox.set('110x200')
        self.formlbl = ttk.Label(self.shot, text='Format')
        self.formlbl.grid(row=1, column=2)
        self.formcbox = ttk.Combobox(self.shot, values=['.npz', 'hdf'], state='readonly')
        self.formcbox.grid(row=1, column=3)
        self.formcbox.set('hdf')
        self.tveclbl = ttk.Label(self.shot, text='Time interval')
        self.tveclbl.grid(row=2, column=1, sticky='W')
        tvecrow = 3
        self.stimevar = tk.StringVar(value='40')
#        self.stimevar = tk.StringVar(value='44.53')
        self.stimelbl = ttk.Label(self.shot, text='from:')
        self.stimeent = ttk.Entry(self.shot, textvariable=self.stimevar)
        self.etimevar = tk.StringVar(value='55')
#        self.etimevar = tk.StringVar(value='44.7')
        self.etimelbl = ttk.Label(self.shot, text='to:')
        self.etimeent = ttk.Entry(self.shot, textvariable=self.etimevar)
        self.stimelbl.grid(row=tvecrow, column=0, sticky='E')
        self.stimeent.grid(row=tvecrow, column=1, sticky='WE')
        self.etimelbl.grid(row=tvecrow, column=2, sticky='E')
        self.etimeent.grid(row=tvecrow, column=3, sticky='WE')
        self.typeframe = ttk.Frame(self, padding=(3, 3, 12, 12),
                                   relief='solid')
        self.typeframe.grid(row=1, column=0, sticky='NSWE')
        t = 0
        l = 0
        # TODO set default values for diagnostics here
        self.bolvar = tk.BooleanVar(value=1)
        self.ntrvar = tk.BooleanVar(value=0)
        self.sxrvar = tk.BooleanVar(value=0)
        self.t0 = ttk.Label(self.typeframe, text='Load:')

        self.chbbol = ttk.Checkbutton(self.typeframe, variable=self.bolvar,
                                      onvalue=True, text='Bolometers')

        self.chbntr = ttk.Checkbutton(self.typeframe, variable=self.ntrvar,
                                      onvalue=True, text='Neutrons')

        self.chbsxr = ttk.Checkbutton(self.typeframe, variable=self.sxrvar,
                                      onvalue=True, text='SXR')
        self.t0.grid(row=t, column=l, sticky=(tk.E, tk.W))
        self.chbbol.grid(row=t + 1, column=l + 0, sticky=(tk.E, tk.W))
        self.chbntr.grid(row=t + 2, column=l + 0, sticky=(tk.E, tk.W))
        self.chbsxr.grid(row=t + 3, column=l + 0, sticky=(tk.E, tk.W))
        #        for i in range(3):
        #            self.columnconfigure(i, weight = 1)

        # data processing frame
        dprocrow = 1
        dproccol = 1
        dprocspan = 1
        self.dprocframe = ttk.Frame(self, padding=(3, 3, 12, 12),
                                    relief='solid')
        self.dprocframe.grid(row=dprocrow, column=dproccol,
                             sticky='NSWE',
                             columnspan=dprocspan)
        self.dprocframe.columnconfigure(0, weight=1)
        self.dprocframe.columnconfigure(1, weight=1)
        self.dproclbl = ttk.Label(self.dprocframe, text='Data processing:')
        self.dproclbl.grid(row=0, column=0, sticky='W')
        self.rgbvar = tk.BooleanVar(value=0)
        self.pwrvar = tk.BooleanVar(value=1)
        self.recvar = tk.BooleanVar(value=0)
        self.prfvar = tk.BooleanVar(value=0)
        self.cntvar = tk.BooleanVar(value=0)
        self.rgbchb = ttk.Checkbutton(self.dprocframe, text='CMY plot',
                                      variable=self.rgbvar, onvalue=True)
        self.rgbchb.grid(row=1, column=0, sticky='W')
        self.pwrchb = ttk.Checkbutton(self.dprocframe, text='Power plot',
                                      variable=self.pwrvar, onvalue=True)
        self.pwrchb.grid(row=2, column=0, sticky='W')
        self.recchb = ttk.Checkbutton(self.dprocframe, text='Rectangles',
                                      variable=self.recvar, onvalue=True)
        self.recchb.grid(row=3, column=0, sticky='W')
        self.prfchb = ttk.Checkbutton(self.dprocframe, text='Profiler',
                                      variable=self.prfvar, onvalue=True)
        self.prfchb.grid(row=4, column=0, sticky='W')
        self.cntchb = ttk.Checkbutton(self.dprocframe, text='Contours',
                                      variable=self.cntvar, onvalue=True)
        self.cntchb.grid(row=5, column=0, sticky='W')


class ExpandMenu(ttk.Frame):
    """
    Executed after main 'Do' method.
    Makes buttons for recreating closed graph windows.
    """

    def __init__(self, parent, **kw):
        ttk.Frame.__init__(self, **kw)

        self.explbl = ttk.Label(self, text='Create window:')
        self.explbl.grid(row=0, column=0, columnspan=2, sticky='W')
        #        for i in range(6):
        #            self.columnconfigure(i, weight = 1)
        bwid = 7
        self.bolbtn = ttk.Button(self, text='bol', width=bwid,
                                 command=parent.make_bol)
        self.bolbtn.grid(row=1, column=0)
        if not parent.menu.bolvar.get():
            self.bolbtn.state(['disabled'])
        self.ntrbtn = ttk.Button(self, text='ntr', width=bwid,
                                 command=parent.make_ntr)
        self.ntrbtn.grid(row=1, column=1)
        if not parent.menu.ntrvar.get():
            self.ntrbtn.state(['disabled'])
        self.sxrbtn = ttk.Button(self, text='sxr', width=bwid,
                                 command=parent.make_sxr)
        self.sxrbtn.grid(row=1, column=2)
        if not parent.menu.sxrvar.get():
            self.sxrbtn.state(['disabled'])
        self.mrfbtn = ttk.Button(self, text='mrf', width=bwid,
                                 command=parent.make_mrf)
        self.mrfbtn.grid(row=1, column=4)
        self.rgbbtn = ttk.Button(self, text='rgb', width=bwid,
                                 command=parent.make_rgb)
        self.rgbbtn.grid(row=2, column=0)
        self.pwrbtn = ttk.Button(self, text='pwr', width=bwid,
                                 command=parent.make_pwr)
        self.pwrbtn.grid(row=2, column=1)
        self.recbtn = ttk.Button(self, text='rec', width=bwid,
                                 command=parent.make_rec)
        self.recbtn.grid(row=2, column=2)
        self.prfbtn = ttk.Button(self, text='prf', width=bwid,
                                 command=parent.make_prf)
        self.prfbtn.grid(row=2, column=3)
        self.cntbtn = ttk.Button(self, text='cnt', width=bwid,
                                 command=parent.make_cnt)
        self.cntbtn.grid(row=2, column=4)
        self.masterbtn = ttk.Button(self, text='master',
                                    command=parent.make_mc, width=bwid + 3)
        self.masterbtn.grid(row=3, column=0, columnspan=2, sticky='W')


class SelectorMenu(ttk.Frame):
    def __init__(self, parent=None):
        ttk.Frame.__init__(self, parent)  # padding = (3,3,3,3))
        self.shot = ttk.Frame(self, padding=(3, 3, 12, 12), relief='solid')
        self.shot.grid(row=0, column=0, columnspan=2)
        tvecrow = 0
        self.tveclbl = ttk.Label(self.shot, text='Time interval')
        self.tveclbl.grid(row=tvecrow, column=1, sticky='W')
        self.stimevar = tk.StringVar(value='0')
        self.stimelbl = ttk.Label(self.shot, text='from:')
        self.stimeent = ttk.Entry(self.shot, textvariable=self.stimevar)
        self.etimevar = tk.StringVar(value='2')
        self.etimelbl = ttk.Label(self.shot, text='to:')
        self.etimeent = ttk.Entry(self.shot, textvariable=self.etimevar)
        self.stimelbl.grid(row=tvecrow+1, column=0, sticky='E')
        self.stimeent.grid(row=tvecrow+1, column=1, sticky='WE')
        self.etimelbl.grid(row=tvecrow+1, column=2, sticky='E')
        self.etimeent.grid(row=tvecrow+1, column=3, sticky='WE')
        self.typeframe = ttk.Frame(self, padding=(3, 3, 12, 12),
                                   relief='solid')
        self.typeframe.grid(row=1, column=0, sticky='NSWE')
        t = 0
        l = 0
        self.var1 = tk.StringVar()
        self.var2 = tk.StringVar()
        self.var3 = tk.StringVar()
        self.varm = tk.StringVar()

        self.t0 = ttk.Label(self.typeframe, text='Load:')
        self.mflbl = ttk.Label(self.typeframe, text='Magnetics:')

        self.ent1 = ttk.Entry(self.typeframe, textvariable=self.var1)
        self.ent2 = ttk.Entry(self.typeframe, textvariable=self.var2)
        self.ent3 = ttk.Entry(self.typeframe, textvariable=self.var3)
        self.entm = ttk.Entry(self.typeframe, textvariable=self.varm)

        self.txt1 = ttk.Button(self.typeframe, text='sel', command=parent.select1, width=3)
        self.txt2 = ttk.Button(self.typeframe, text='sel', command=parent.select2, width=3)
        self.txt3 = ttk.Button(self.typeframe, text='sel', command=parent.select3, width=3)
        self.txtm = ttk.Button(self.typeframe, text='sel', command=parent.selectm, width=3)

        self.t0.grid(row=t, column=l, sticky=(tk.E, tk.W))

        self.ent1.grid(row=t + 1, column=l + 0, sticky=(tk.E, tk.W))
        self.ent2.grid(row=t + 2, column=l + 0, sticky=(tk.E, tk.W))
        self.ent3.grid(row=t + 3, column=l + 0, sticky=(tk.E, tk.W))

        self.txt1.grid(row=t + 1, column=l + 1, sticky=(tk.E, tk.W))
        self.txt2.grid(row=t + 2, column=l + 1, sticky=(tk.E, tk.W))
        self.txt3.grid(row=t + 3, column=l + 1, sticky=(tk.E, tk.W))

        self.mflbl.grid(row=t + 4, column=l, sticky=(tk.E, tk.W))
        self.entm.grid(row=t + 5, column=l + 0, sticky=(tk.E, tk.W))
        self.txtm.grid(row=t + 5, column=l + 1, sticky=(tk.E, tk.W))
        #        for i in range(3):
        #            self.columnconfigure(i, weight = 1)

        # data processing frame
        dprocrow = 1
        dproccol = 1
        dprocspan = 1
        self.dprocframe = ttk.Frame(self, padding=(3, 3, 12, 12),
                                    relief='solid')
        self.dprocframe.grid(row=dprocrow, column=dproccol,
                             sticky='NSWE',
                             columnspan=dprocspan)
        self.dprocframe.columnconfigure(0, weight=1)
        self.dprocframe.columnconfigure(1, weight=1)
        self.dproclbl = ttk.Label(self.dprocframe, text='Data processing:')
        self.dproclbl.grid(row=0, column=0, sticky='W')
        self.rgbvar = tk.BooleanVar(value=0)
        self.pwrvar = tk.BooleanVar(value=1)
        self.recvar = tk.BooleanVar(value=0)
        self.prfvar = tk.BooleanVar(value=0)
        self.cntvar = tk.BooleanVar(value=0)
        self.rgbchb = ttk.Checkbutton(self.dprocframe, text='CMY plot',
                                      variable=self.rgbvar, onvalue=True)
        self.rgbchb.grid(row=1, column=0, sticky='W')
        self.pwrchb = ttk.Checkbutton(self.dprocframe, text='Power plot',
                                      variable=self.pwrvar, onvalue=True)
        self.pwrchb.grid(row=2, column=0, sticky='W')
        self.recchb = ttk.Checkbutton(self.dprocframe, text='Rectangles',
                                      variable=self.recvar, onvalue=True)
        self.recchb.grid(row=3, column=0, sticky='W')
        self.prfchb = ttk.Checkbutton(self.dprocframe, text='Profiler',
                                      variable=self.prfvar, onvalue=True)
        self.prfchb.grid(row=4, column=0, sticky='W')
        self.cntchb = ttk.Checkbutton(self.dprocframe, text='Contours',
                                      variable=self.cntvar, onvalue=True)
        self.cntchb.grid(row=5, column=0, sticky='W')
        return
