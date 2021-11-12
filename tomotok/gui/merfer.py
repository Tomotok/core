# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
MEasured vs RetroFit comparator graphical window widget
"""
from pathlib import Path

import numpy as np

from .widgets import GraphWindow


class GraphMerfer(GraphWindow):
    """
    Widgets for comparing MEasured data with RetroFit
    
    Parameters
    ----------
    parent : TKInter widget
        Default is None.
    data : dict
        Contains data, default is empty dict.
    """
    def __init__(self, parent=None, data=None, errors=None, **kw):
        GraphWindow.__init__(self, parent, **kw)
        if data is None:
            data = {}
        if 'title' in kw:
            self.title(kw['title'])
        else:
            self.title('Merfer')
        self.colors = {'bol': 'r', 'ntr': 'g', 'sxr': 'b'}
        # self.colors = ('c', 'm', 'y')
        self.update_data(data, err=errors)

    def update_data(self, ndata, err=None, **kw):
        """
        Loads new data.
        
        Parameters
        ----------
        ndata : dict
            New data to be loaded to Graph window
        err : np.ndarray
            contains errors of measured data with shape (#chnl, #tslices)
        """
#        self.data = {}
        if err is None:
            err = {'bol': None, 'ntr': None, 'sxr': None}
        self.errors = err
        self.tvecs = {}
        self.diags = []
        try:
            self.shot = ndata['shot']
        except KeyError:
            self.shot = 0
        keys = ndata.keys()
        self.me_data = {}
        self.rf_data = {}
        if 'tind' not in keys:
            ndata['tind'] = {}
        if 'bol' in keys:
            if 'bol' in ndata['tind']:
                tind = ndata['tind']['bol']
            else:
                tind = range(len(ndata['bol']['tvec']))
            me_bol = ndata['bol']['meli'][..., tind]
            max_ = np.max(me_bol, axis=0)
            me_bol = me_bol / max_
            try:
                self.errors['bol'] = ndata['errors']['bol'] / max_
            except KeyError:
                self.errors['bol'] = None
            self.me_data['bol'] = me_bol
            rf_bol = ndata['bol']['bcli'][..., tind]
            rf_bol = rf_bol / np.max(rf_bol, axis=0)
            self.rf_data['bol'] = rf_bol
            self.tvecs['bol'] = ndata['bol']['tvec'][tind]
            self.diags.append('bol')
        if 'ntr' in keys:
            if 'ntr' in ndata['tind']:
                tind = ndata['tind']['ntr']
            else:
                tind = range(len(ndata['ntr']['tvec']))
            me_ntr = ndata['ntr']['meli'][..., tind]
            max_ = np.max(me_ntr, axis=0)
            me_ntr = me_ntr / max_
            try:
                self.errors['ntr'] = ndata['errors']['ntr'] / max_
            except KeyError:
                self.errors['ntr'] = None
            self.me_data['ntr'] = me_ntr
            rf_ntr = ndata['ntr']['bcli'][..., tind]
            rf_ntr = rf_ntr / np.max(rf_ntr, axis=0)
            self.rf_data['ntr'] = rf_ntr
            self.tvecs['ntr'] = ndata['ntr']['tvec'][tind]
            self.diags.append('ntr')
        if 'sxr' in keys:
            if 'sxr' in ndata['tind']:
                tind = ndata['tind']['sxr']
            else:
                tind = range(len(ndata['sxr']['tvec']))
            me_sxr = ndata['sxr']['meli'][..., tind]
            max_ = np.max(me_sxr, axis=0)
            me_sxr = me_sxr / max_
            try:
                self.errors['sxr'] = ndata['errors']['sxr'] / max_
            except KeyError:
                self.errors['sxr'] = None
            self.me_data['sxr'] = me_sxr
            rf_sxr = ndata['sxr']['bcli'][..., tind]
            rf_sxr = rf_sxr / np.max(rf_sxr, axis=0)
            self.rf_data['sxr'] = rf_sxr
            self.tvecs['sxr'] = ndata['sxr']['tvec'][tind]
            self.diags.append('sxr')
        self.key = self.diags[0]
        tvec = np.array(self.tvecs[self.key])
        self.set_vals(vals=tvec)
        self.plot(0)
    
    def plot(self, val, *args):
        self.ax.cla()
        for d in self.diags:
            tind = np.argmin(np.abs(self.tvecs[d] - self.vals[val]))
            chnls = np.arange(self.me_data[d].shape[0], dtype=np.float64)
            # self.ax.plot(self.me_data[d][:, tind], c='C{}'.format(i), label='Measured {}'.format(d))
            try:
                err = self.errors[d][:, tind]
            except TypeError:
                err = None
            self.ax.errorbar(chnls, self.me_data[d][:, tind], err, c=self.colors[d],
                             ls='', marker='+', capsize=3, label='Measured {}'.format(d))
            self.ax.plot(self.rf_data[d][:, tind], c=self.colors[d], alpha=0.8, label=r'Retrofit {}'.format(d))
            # sc = np.zeros(chnls.size + 2)
            # sc[0] = chnls[0] - 0.5
            # sc[-1] = chnls[-1] + 0.5
            # sc = chnls
            # self.ax.step(sc, self.rf_data[d][:, tind], c=self.colors[d],
            #              alpha=0.6, where='mid', label=r'Retrofit {}'.format(d))
            # self.ax.bar(chnls, self.rf_data[d][:, tind], edgecolor=self.colors[d], color='',
            #             label=r'Retrofit {}'.format(d))
        self.ax.legend()
        self.ax.set_xlabel('Channel')
        self.ax.set_ylabel('Normalized Power [-]')
        self.gcanvas.draw()
    
    def save(self, fname='Graphs'):
        val = self.scale.get()
        tslice = self.vals[val] * 1000
        # form = 'eps'
        form = 'png'
        name = '{} MERFer at {:.0f}ms.{}'.format(self.shot, tslice, form)
        floc = Path() / fname / name
        if not floc.parent.exists():
            floc.parent.mkdir(parents=True)
        self.fig.savefig(floc)
