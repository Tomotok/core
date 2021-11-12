# Copyright 2021 Institute of Plasma Physics of the Czech Academy of Sciences. 
#
# Licensed under the EUPL-1.2 or later.
"""
WIP

Contains class that does data filtering and stuff. It can also handle error estimation.
"""
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import generic_filter1d
from scipy.interpolate import interp1d
from scipy.signal import medfilt, butter, lfilter
from scipy.stats import sem


class DataHandler(object):
    """
    Does filtering and stuff

    Attributes
    ----------
    chnls : np.ndarray
    time : np.ndarray
    data : np.ndarray
    errors : np.ndarray
    """

    def __init__(self, data, min_error=0):
        """

        Parameters
        ----------
        data : dict
            [description]
        min_error : float, optional
            [description], by default 0
        """
        self.chnls = data['chnl']
        self.time = data['time']
        self.data = data['values']
        self._original = data.copy()
        self.errors = np.zeros_like(self.data) + min_error
        return

    # @property
    # def errors(self):
    #     return self._errors

    # @errors.setter
    # def errors(self, err):
    #     try:
    #         assert err.shape == self.data.shape
    #         self.errors = err
    #     except AssertionError:
    #         raise ValueError('Errors must have same shape as data')

    def reset(self, min_error=0):
        self.data = self._original
        self.errors = np.zeros_like(self.data) + min_error

    def add_errors(self, err):
        """
        Parameters
        ----------
        err : 
        """
        self.errors += err
        # self.errors = np.sqrt(np.power(self.err, 2) + np.power(err, 2))

    def errors_scalar_multiply(self, scalar):
        try:
            assert isinstance(scalar, (int, float))
            self.errors *= scalar
        except AssertionError:
            raise ValueError('scalar must be instance of int or float')

    def median_filter(self, kernel_size=1, error_type='sem'):
        if kernel_size % 2 == 0:
            warn('Medfilt does not support even values of kernel size. Adding +1.')
            kernel_size += 1
        self._filtering_errors(error_type, kernel_size=kernel_size)
        self._broadcast_to_channels(medfilt, kernel_size=kernel_size)

    def average(self, avg_num, tvec):
        tvec = np.array(tvec, ndmin=1)
        data = self.data
        avg_data = np.zeros((tvec.size, self.chnls.size))
        stds = np.zeros_like(avg_data)
        for i in range(tvec.size):
            idx = np.searchsorted(self.time, tvec[i])
            start = max(0, idx - avg_num)
            dsel = data[idx:start:-1, :]
            avg_data[i, ...] = dsel.mean(axis=0)
            stds[i, ...] = dsel.std(axis=0)
        return avg_data, stds

    # def bpf(self, error_type='coef', freq_lim=None):
    #     kwargs = {
    #         'lowcut': self.params['filter'][1],
    #         'highcut': self.params['filter'][2],
    #         'fs': 1 / (np.mean(np.diff(tvec))),
    #         'order': 2}
    #
    #     data_cut = self.cut_data(data_raw)  # TODO
    #
    #     data = self.__iterate(data_cut, self.butter_bandpass_filter, kwargs)
    #
    #     errors = self.compute_data_errors(data, error_type)
    #
    #     data_out = self.interpolate(data)
    #
    #     return data_out, errors

    # def gauss(self, error_type='sem', data_smooth=0):
    #     kwargs = {
    #         'sigma': data_smooth,
    #         'mode': 'nearest',
    #     }
    #
    #     self._broadcast_to_channels(self.butter_bandpass_filter, kwargs)
    #
    #     errors = self.compute_data_errors(error_type)

    def _interpolate_data(self, tvec):
        """
        Interpolates data to given time vector

        Parameters
        ----------
        tvec : np.ndarray

        Returns
        -------
        ndarray
        """
        data_interp = np.zeros((tvec.size, self.chnls.size))
        for i in range(self.chnls.size):
            dataf = interp1d(self.time, self.data[:, i])
            data_interp[:, i] = dataf(tvec)
        return data_interp

    def _interpolate_errors(self, tvec):
        """
        Interpolates data to given time vector

        Parameters
        ----------
        tvec : np.ndarray

        Returns
        -------
        ndarray
        """
        err_interp = np.zeros((tvec.size, self.chnls.size))
        for i in range(self.chnls.size):
            errf = interp1d(self.time, self.errors[:, i])
            err_interp[:, i] = errf(tvec)
        return err_interp

    def interpolate(self, tvec):
        """
        Interpolates both data and errors to given time vector.

        Parameters
        ----------
        tvec : int, float, np.ndarray

        Returns
        -------
        data, err : ndarray
        """
        tvec = np.array(tvec, ndmin=1)
        if tvec.min() < self.time.min() or tvec.max() > self.time.max():
            raise ValueError('Interpolating outside data time interval.')
        data = self._interpolate_data(tvec)
        err = self._interpolate_errors(tvec)
        return data, err

    def _broadcast_to_channels(self, func, *args, **kwargs):
        """
        Modifies data by applying provided function to each channel

        Parameters
        ----------
        func : callable
        args
            positional arguments
        kwargs
            keywords arguments
        """
        for i in range(self.data.chnl.size):
            self.data[i, :] = func(self.data[i, :], *args, **kwargs)

    @staticmethod
    def _sem(iline, oline, width):
        """
        Static wrapper for generic_filter1d using scypy.stats.sem as filter function over width samples
        """
        for i in range(len(oline)):
            oline[i] = sem(iline[i:i + width])

    @staticmethod
    def _std(iline, oline, width):
        """
        Static wrapper for generic_filter1d using np.std as filter function over width samples
        """
        for i in range(len(oline)):
            oline[i] = np.std(iline[i:i + width])

    @staticmethod
    def _none(iline, oline):
        pass

    def _error_filter(self, func, kernel_size, **kwargs):
        errors = generic_filter1d(self.data, func, kernel_size, axis=1, **kwargs)
        self.add_errors(errors)

    # TODO reconsider explicit error computation
    # def error_std(self, kernel_size):
    #     errors = generic_filter1d(self.data, self._std, kernel_size, axis=1, extra_keywords={'width': kernel_size})
    #     # self.add_errors(errors)
    #     return errors
    #
    # def error_sem(self, kernel_size):
    #     errors = generic_filter1d(self.data, self._std, kernel_size, axis=1, extra_keywords={'width': kernel_size})
    #     # self.add_errors(errors)
    #     return errors

    def _filtering_errors(self, error_type='std', **kwargs):
        """
        Computes errors induced by filtering data.
        """
        size = 1
        err_kw = {}
        if error_type.lower() == 'std':
            try:
                size = kwargs['kernel_size']
            except KeyError:
                raise ValueError('Parameter `kernel_size` must be provided for std method')
            error_func = self._std
            err_kw = {'extra_keywords': {'width': size}}
        elif error_type.lower() == 'sem':
            try:
                size = kwargs['kernel_size']
            except KeyError:
                raise ValueError('Parameter `kernel_size` must be provided for sem method')
            err_kw = {'extra_keywords': {'width': size}}
            error_func = self._sem
        elif error_type.lower() == 'poisson':
            raise NotImplementedError()
        elif error_type.lower() == 'wo':
            error_func = self._none
        else:
            raise ValueError('Unsuported error_type')
        self._error_filter(error_func, kernel_size=size, **err_kw)

    def crop_data(self, tlim, extend_num=None):
        """
        Cuts data to requested interval

        Parameters
        ----------
        tlim : tuple
            contains limits of requested time axis
        extend_num : int, optional
            contains number of points outside tlim for filtering methods
            (default value is calculated to fit chosen filtering method) <- should be determined by user
        """
        if tlim[0] < self.time.min() or self.time.max() < tlim[1]:
            raise ValueError('Requested time range out of data time vector.')
        t0 = np.searchsorted(self.time, tlim[0])
        t1 = np.searchsorted(self.time, tlim[0], side='right')
        if extend_num is None:
            self.data = self.data[t0:t1, :]
            self.errors = self.errors[t0:t1, :]
        else:
            raise NotImplementedError('Extension of croping interval was not implemented.')
            # FIXME make the method work with data attribute, remove return statement
            # TODO reconsider extend_num
            # ind_t0 = np.where(self.tvec >= tlim[0])[0][0]
            # ind_t1 = np.where(self.tvec <= tlim[1])[0][-1]
            #
            # try:
            #     self.data = self.data[:, ind_t0 - extend_num:ind_t1 + extend_num + 1]
            # except IndexError:
            #     warn('Time range with extension is out of data time vector. Using maximal possible extension.')
            #     self.data = self.data[:, max(0, ind_t0 - extend_num):min(ind_t1 + extend_num + 1, len(data.t))]
            # if ind_t0 - extend_num < 0:
            #     n_add = extend_num - ind_t0
            #     data_start = np.matlib.repmat(data.values[:, 0], n_add, 1).T
            #     self.data.values = np.concatenate((data_start, data.values), axis=1)
            #     dt = data.t.values[ind_t0 + 1] - data.t.values[ind_t0]
            #     self.data.t.values = np.concatenate(
            #         (np.linspace(data.t.values[0] - n_add * dt, data.t.values[0] - dt, n_add), data.t.values))
            #
            # if ind_t1 + extend_num > len(data.t):
            #     n_add = extend_num + ind_t1 - data.shape[1] + 1
            #     data_end = np.matlib.repmat(data.values[:, -1], n_add, 1).T
            #     self.data.values = np.concatenate((data.values, data_end), axis=1)
            #     dt = data.t.values[ind_t1] - data.t.values[ind_t1 - 1]
            #     self.data.t.values = np.concatenate(
            #         (data.t.values, np.linspace(data.t.values[-1] + dt, data.t.values[-1] + dt * n_add), n_add))

    def downsample(self, n, average=False):
        """
        Keeps only each n-th time slice of data.

        Parameters
        ----------
        n : int
        average : bool, optional
            not implemented yet
        """
        if average:
            raise NotImplementedError()
        else:
            self.data = self.data[::n, :]
            self.errors = self.errors[::n, :]

    def butter_bandpass_filter(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, self.data)
        return y

    def remove_nan_chnls(self):
        """
        Removes channels that contain at least one Nan value.
        """
        mask = np.isfinite(self.data).all(0)
        self.chnls = self.chnls[mask]
        self.data = self.data[:, mask]
        self.errors = self.errors[:, mask]

    def remove_chnls(self, chnls):
        """
        Removes specified channels from data

        Parameters
        ----------
        chnls : iterable of ints
        """
        # chnls = np.array(chnls, ndmin=1, dtype=np.int)
        chnls = np.unique(chnls)
        mask = np.isin(chnls, self.chnls)
        if not mask.all():
            warn('Channels not in data: {}'.format(chnls[~mask]))
        chnls = chnls[mask]
        idx = np.searchsorted(self.chnls, chnls)
        mask = np.ones_like(self.chnls, dtype=np.bool)
        mask[idx] = 0
        self.chnls = self.chnls[mask]
        self.data = self.data[:, mask]
        self.errors = self.errors[:, mask]

    def remove_zero_chnls(self):
        sm = self.data.sum(axis=0)  # along time axis
        mask = sm != 0
        self.chnls = self.chnls[mask]
        self.data = self.data[:, mask]
        self.errors = self.errors[:, mask]

    def remove_below(self, threshold):
        """
        Removes channels with average signal bellow given threshold

        Parameters
        ----------
        threshold : float
        """
        sm = self.data.sum(axis=0)
        mask = sm >= threshold * self.time.size
        self.chnls = self.chnls[mask]
        self.data = self.data[:, mask]
        self.errors = self.errors[:, mask]

    def fourier(self, freqs):
        """
        Transforms data to frequency domain using fourier transform

        Parameters
        ----------
        freqs : list
            values of frequencies used for new `time vector`

        Returns
        -------
        DataHandler
            new instance of DataHandler with transformed data and errors
        """
        raise NotImplementedError()

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def __repr__(self):
        msg = '<DataHandler>'
        msg += '\n' + self.chnls.__repr__() + self.time.__repr__()
        msg += '\n Data \n' + '\n'.join(self.data.__repr__())#.split('\n')[1:-3])
        msg += '\n Errors \n' + '\n'.join(self.errors.__repr__())#.split('\n')[1:-3])
        return msg

    # def sel(self, *args, **kwargs):
    #     # for i in ('chnl', 't'):
    #     #     try:
    #     #         indexers[i] = np.array(indexers[i], ndmin=1)
    #     #     except KeyError:
    #     #         pass
    #     self._data = self.data.sel(*args, **kwargs)
    #     self._errors = self.errors.sel(*args, **kwargs)
    #     # dh = DataHandler(data)
    #     # dh.errors = err
    #     # return dh

    # def isel(self, *args, **indexers):
    #     data = self.data.isel(*args, **indexers)
    #     err = self.errors.isel(*args, **indexers)
    #     dh = DataHandler(data)
    #     dh.errors = err
    #     return dh

    def plot_chnls(self, chnls=None):
        if chnls is None:
            chnls = self.chnls
        fig = plt.figure()
        for ch in chnls:
            data = self.data[:, ch]
            err = self.errors[:, ch]
            plt.fill_between(self.time, data - err, data + err, alpha=0.5, interpolate=True)
            plt.plot(self.time, data)
        return fig

    def plot_time(self, time=0):
        fig = plt.figure()
        tidx = np.searchsorted(self.time, time)
        data = self.data[tidx, :]
        err = self.errors[tidx, :]
        plt.errorbar(self.chnls, data, err, ls='', marker='+', capsize=3, c='k', alpha=1)
        plt.bar(self.chnls, data)
        return fig
