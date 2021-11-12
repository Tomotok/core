# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
Library of custom tkinter widgets that have not been specially designed for displaying or processing tomography results.
"""
import sys
import tkinter as tk
from tkinter import ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ConsoleText(tk.Frame):
    """
    A Tkinter Text widget that provides a scrolling display of console stdout.
    """

    def __init__(self, master=None, cnf=None, **kw):
        """
        Parameters
        ----------
        master
        cnf
        kw

        Notes
        -----
        See the __init__ for Tkinter.Text for most of this stuff.
        """
        tk.Frame.__init__(self, master)
        if cnf is None:
            cnf = {}
        self.varscrl = tk.BooleanVar(value=True)
        try:
            self.varscrl.set(kw['scroll'])
        except KeyError:
            self.varscrl.set(True)
        self.txt = tk.Text(self, cnf, **kw)
        self.std_orig = sys.stdout
        self.write_orig = sys.stdout.write
        self.sbar = ttk.Scrollbar(self, orient=tk.VERTICAL,
                                  command=self.txt.yview)
        self.txt['yscrollcommand'] = self.sbar.set
        self.txt['state'] = 'disabled'
        self.txt.grid(row=0, column=0, sticky='NSEW')
        self.sbar.grid(row=0, column=1, sticky='NSW')
        self.scrollcb = ttk.Checkbutton(self, onvalue=True,
                                        command=self.set_scrl,
                                        variable=self.varscrl)
        self.scrollcb.grid(row=1, column=1, sticky='W')
        self.scrlbl = ttk.Label(self, text='Scroll')
        self.scrlbl.grid(row=1, column=0, sticky='E')
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.columnconfigure(0, weight=1)
        self.running = False
        self.scroll = False

    def start(self):
        sys.stdout.write = self.write
        self.running = True

    #        self.log_update()

    def stop(self):
        sys.stdout.write = self.write_orig
        self.running = False

    def set_scrl(self):
        self.scroll = self.varscrl.get()

    def write(self, val):
        self.txt['state'] = 'normal'
        self.txt.insert('end', val)
        if self.varscrl.get():
            self.txt.see('end')
        self.txt['state'] = 'disabled'
        self.update()

    def log_update(self):
        self.update()
        # print('log_up', self.running, flush = True)
        if self.running:
            # print('after called')
            self.after(1000, self.log_update)


class ConsoleWindow(tk.Toplevel):
    """
    Console text embedded in Toplevel window.
    """

    def __init__(self, parent=None, **kw):
        tk.Toplevel.__init__(self, parent, **kw)
        if 'title' in kw.keys():
            self.title(kw['title'])
        else:
            self.title('Log window')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.console = ConsoleText(self)
        self.console.grid(row=0, column=0, sticky='NSWE')

    def start(self):
        self.console.start()

    def stop(self):
        self.console.stop()


class GraphWindow(tk.Toplevel):
    """
    Basic window prepared for displaing data sequence with use of matplotlib.
    Contains Figure and Axes for plotting, but no method for plotting or 
    atribute with data.
    By default it consists of listbox with data sequence, frame for buttons,
    canvas for plot and slider for navigation.
    
    Parameters
    ----------
    parent : TKInter widget
        Default is None. Used in GUI interface to specify hierarchy.
    
    Keywords
    --------
    vals : array-like
        Contains data from time vector to be shown in listbox and
        under slider.
    
    slices : int
        Number of time steps. Use ether vals or slices. Use if no time 
        vector data are available. Creates vals as range(slices).
    
    slavechbutton : bool
        Switches whether slave check button will be created. If True a button
        is created, that if unchecked will switch off control from master.
    """
    def __init__(self, parent=None, **kw):
        tk.Toplevel.__init__(self, parent)
        self.bind('<Left>', self.go_prev)
        self.bind('<Right>', self.go_next)
        self.bind('s', self.save)
        self.data = []
        self.grid()
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.slices = 0
        self.vals = []
        self.init_datarange(**kw)

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.gcanvas = FigureCanvasTkAgg(self.fig, master=self)
        self.gcanvas.get_tk_widget().grid(row=0, column=1, sticky='NSW')
        self.gcanvas._tkcanvas.grid(row=0, column=1, sticky='NSWE')

        self.lboxmenu = ttk.Frame(self)
        self.lboxmenu.grid(row=0, column=0, sticky='NSWE', rowspan=2)
        self.lboxmenu.rowconfigure(0, weight=0)
        self.lboxmenu.rowconfigure(1, weight=1)
        self.lboxmenu.rowconfigure(2, weight=0)
        self.lboxlbl = ttk.Label(self.lboxmenu, text='Data slices')
        self.lboxlbl.grid(row=0, column=0, sticky='NSWE')
        self.lbvar = tk.StringVar()
        self.lbvar.set([i for i in self.vals])
        self.lbox = tk.Listbox(self.lboxmenu, width=5)
        self.lbox['listvariable'] = self.lbvar
        self.sbar = ttk.Scrollbar(self.lboxmenu, orient=tk.VERTICAL,
                                  command=self.lbox.yview)
        self.lbox['yscrollcommand'] = self.sbar.set
        self.lbox.activate(0)
        self.lbox.bind('<Double-Button-1>', self.lbox_selection)
        self.lbox.bind('<Return>', self.lbox_selection)
        self.lbox.grid(row=1, column=0, sticky='NSWE')
        self.sbar.grid(row=1, column=1, sticky='NSW')
        self.bb_row = 0
        self.bbox = ttk.Frame(self.lboxmenu)
        self.bbox.grid(row=2, column=0, sticky='NSWE')
        self.but = ttk.Button(self.bbox, text='Save',
                              command=self.save)
        self.but.grid(row=self.bb_row, column=0, sticky='N')
        self.bb_row_raise()
        try:
            assert kw['slavechb'] is True
            self.slave = tk.BooleanVar()
            self.slave.set(True)
            slavechb = ttk.Checkbutton(master=self.bbox, text='Slave', variable=self.slave, onvalue=True)
            self.slavechb = slavechb
            self.slavechb.grid(row=self.bb_row, column=0, sticky='W')
            self.bb_row_raise()
        except (KeyError, AssertionError):
            self.slave = tk.BooleanVar(value=True)

        self.slider = ttk.Frame(self)
        self.slider.columnconfigure(1, weight=1)
        self.slider.rowconfigure(0, weight=0)
        self.slider.rowconfigure(1, weight=0)
        self.slider.grid(sticky='NSWE')
        self.scale = tk.Scale(self.slider, orient=tk.HORIZONTAL,
                              showvalue=False,
                              from_=0, to=self.slices - 1,
                              command=self.slider_move,
                              )
        self.label = ttk.Label(self.slider, text='start')
        self.next = ttk.Button(self.slider, text='>', width=2,
                               command=self.go_next)
        self.prev = ttk.Button(self.slider, text='<', width=2,
                               command=self.go_prev)
        self.slider.grid(row=1, column=1)
        self.label.grid(row=1, column=1, sticky='N')
        self.scale.grid(row=0, column=1, sticky='WE')
        self.next.grid(row=0, column=2)
        self.prev.grid(row=0, column=0)
        self.label['text'] = self.vals[0]

    def bb_row_raise(self):
        """
        Increases bb_row, atribute that describes number of placed buttons in
        button box frame widget positioned under listbox. Used for automated 
        row placement.
        Should be called after placing new button to bbox.
        """
        self.bb_row += 1

    def init_datarange(self, **kw):
        """
        Prepares vals or slices for widget construction.
        """
        if 'vals' in kw.keys():
            self.vals = kw['vals']
            self.slices = len(self.vals)
        else:
            self.set_slices(**kw)
            self.vals = [i + 1 for i in range(self.slices)]

    def set_slices(self, **kw):
        """
        Checks keywords for slices infromation. Sets slices to specified number
        or 1 if number of slices was not specified.
        """
        if 'slices' in kw.keys():
            val = kw['slices']
        else:
            val = 1
        self.slices = val

    def set_vals(self, **kw):
        """
        Rutine to set values for listbox menu and displayed numbers and range
        of scale widget.
        """
        if 'vals' in kw.keys():
            self.vals = kw['vals']
            self.slices = len(self.vals)
            self.scale['to'] = self.slices - 1
        else:
            self.set_slices(**kw)
            self.scale['to'] = self.slices - 1
            self.vals = [i + 1 for i in range(self.slices)]
        self.lbvar.set([i for i in self.vals])
        self.label['text'] = self.vals[self.scale.get()]

    def go_to(self, val, *args):
        """
        Puts slider into position given by 'val'.
        
        Parameters
        ----------
        val : int
            Nunber of slider position.
        """
        self.scale.set(val)
        return

    def slider_move(self, *args):
        """
        Executed when position of slider is changed. Changes label of slider,
        plots apropirate data and moves in listbox widget.
        """
        num = self.scale.get()
        self.label['text'] = self.vals[num]
        self.lbox.activate(num)
        self.lbox.see(num)
        self.plot(num)
        return

    def go_next(self, *args):
        new = self.scale.get() + 1
        self.go_to(new)
        return

    def go_prev(self, *args):
        new = self.scale.get() - 1
        self.go_to(new)
        return

    def lbox_selection(self, *args):
        """
        Executed when selection in listbox was executed.
        """
        val = self.lbox.curselection()[0]
        self.go_to(val)
        return

    def save(self, *args):
        raise NotImplementedError()

    def update_graph(self):
        """
        Plots data slice apropirate to slider position.
        """
        val = self.scale.get()
        self.plot(val)

    def update_data(self, ndata, **kw):
        """
        Loads new data.
        
        Parameters
        ----------
        ndata : data structure
            New data to be loaded to Graph window.
        """
        self.data = ndata
        self.set_vals(**kw)

    def plot(self, num):
        """

        Parameters
        ----------
        num

        Returns
        -------

        """
        raise NotImplementedError


class MasterControls(tk.Toplevel):
    def __init__(self, parent, slices, *controled, **kw):
        tk.Toplevel.__init__(self, parent)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.title('Master controls')
        self.grid()
        self.ctrl = controled
        self.slices = slices
        self.vals = []
        self.set_vals(**kw)
        self.slider = ttk.Frame(self)
        self.slider.columnconfigure(1, weight=1)
        self.slider.rowconfigure(0, weight=1)
        self.slider.rowconfigure(1, weight=0)
        self.slider.rowconfigure(2, weight=1)
        self.slider.grid(row=0, column=0, sticky='NSWE')
        self.scale = tk.Scale(self.slider, orient=tk.HORIZONTAL,
                              showvalue=False,
                              from_=0, to=slices - 1,
                              command=self.slider_move)
        self.scale.grid(row=1, column=1, sticky='WE')
        self.label = ttk.Label(self.slider, text=self.vals[0])
        self.label.grid(row=2, column=1, sticky='N')
        self.next = ttk.Button(self.slider, text='->',
                               command=self.go_next)
        self.next.grid(row=1, column=2)
        self.prev = ttk.Button(self.slider, text='<-',
                               command=self.go_prev)
        self.prev.grid(row=1, column=0)
        self.syncvar = tk.BooleanVar()
        self.syncvar.set(1)
        sync = ttk.Checkbutton(self.slider, text='Synchronize',
                               #                                   onvalue = True,
                               variable=self.syncvar)
        sync.grid(row=0, columnspan=3, sticky='S')
        self.bind('<Left>', self.go_prev)
        self.bind('<Right>', self.go_next)
        self.bind('<Return>', self.change_sync)
        self.nogo = ttk.Label(self, text='Create graphs to be controled '
                                         'before creating master controls')
        if len(controled) == 0:
            self.nogo.grid(sticky='NSWE')
        else:
            self.nogo.destroy()

    def set_vals(self, **kw):
        if 'vals' in kw.keys():
            self.vals = kw['vals']
        else:
            self.vals = [i + 1 for i in range(self.slices)]

    # def set_ctrl(self, control):
    #     self.ctrl = control

    def set_slices(self, slices, **kw):
        self.slices = slices
        self.set_vals(**kw)
        self.scale['to'] = slices - 1

    def change_sync(self, *args):
        state = self.syncvar.get()
        self.syncvar.set(not state)

    def slider_move(self, *args):
        num = self.scale.get()
        self.label['text'] = self.vals[num]
        for i in self.ctrl:
            try:
                if i.slave.get():
                    i.go_to(num)
            except AttributeError:
                self.ctrl = self.master.get_ctrl()

    def go_next(self, *args):
        if self.syncvar.get():
            new = self.scale.get() + 1
            self.scale.set(new)
        else:
            for i in self.ctrl:
                if i.slave.get():
                    i.go_next()

    def go_prev(self, *args):
        if self.syncvar.get():
            new = self.scale.get() - 1
            self.scale.set(new)
        else:
            for i in self.ctrl:
                if i.slave.get():
                    i.go_prev()
