#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:19:51 2025

@author: rachelwillis
"""

import numpy as np
from scipy.signal import sosfilt
from scipy.signal.windows import hann

import par_file as pf


def AGC_1D(data_in, window):
    """
    Function to apply Automatic Gain Control (AGC) to a single trace.

    Parameters
    ----------
    data_in : 1D numpy array
        Input data
    window : int
        window length in number of samples (not time)

    Returns
    -------
    y : Amplitude normalized input trace

    tosum : AGC scaling values that were applied

    """
    nt = len(data_in)
    y = np.zeros(nt)
    tosum = np.zeros(nt)

    # enforce that window is integer
    if type(window) != int:
        window = int(window)

    # enforce that window is smaller than length of time series
    if window > nt:
        window = nt

    # enforce that window is odd
    if window % 2 == 0:
        window = window - 1

    len2 = int((window-1)/2)

    e = np.cumsum(np.abs(data_in))

    for i in range(nt):
        if i-len2-1 > 0 and i+len2 < nt:
            tosum[i] = e[i+len2] - e[i-len2-1]
        elif i-len2-1 < 1:
            tosum[i] = e[i+len2]
        else:
            tosum[i] = e[-1] - e[i-len2-1]

    y = data_in / tosum
    return y

def single_whiten_taper(data, f_taper, f_smooth=10):
    """
    Function that applies spectral whitening to trace.

    Parameters
    ----------
    data_in : 1D numpy array
        Input data
    f_taper : 1D numpy array
        Frequency representation of bandpass filter to be
        applied after spectral whitening.
    freq_smooth : int
        window length in number of samples for smoothing in
        frequency domain

    Returns
    -------
    double_white : Spectral whitened trace
    """
    data_tapered = data
    spectrum = np.fft.rfft(data_tapered)
    spectrum_white = AGC_1D(spectrum, f_smooth)
    spectrum_white_tapered = spectrum_white * f_taper
    data_white = np.fft.irfft(spectrum_white_tapered)
    return data_white


def preproc_test(tr_id, st_demean, dec_factor=5):
    """Taper, filter, and decimate along time-axis."""
    npts = len(st_demean[:, tr_id])
    wlen = int(pf.max_perc * npts)
    taper_sides = hann(2*wlen+1)
    taper = np.hstack((taper_sides[:wlen], np.ones(npts-2*wlen), taper_sides[len(taper_sides)-wlen:]))
    tapered = st_demean[:, tr_id] * taper
    firstpass = sosfilt(pf.sos, tapered)
    filtered = sosfilt(pf.sos, firstpass[::-1])[::-1]
    downsampled = filtered[::pf.dec_factor]
    filtered_white = single_whiten_taper(downsampled, f_taper=pf.f_taper, f_smooth=10)
    return filtered_white


def preproc_x(tr_t_id, st_whitened_agc):
    """Taper, filter, and decimate along space-axis."""
    tapered = st_whitened_agc[tr_t_id, :] * pf.taper_x
    firstpass = sosfilt(pf.sos_x, tapered)
    filtered = sosfilt(pf.sos_x, firstpass[::-1])[::-1]
    return filtered

def AGC(data_in, window, normalize=False, time_axis='vertical'):
    """
    Function to apply Automatic Gain Control (AGC) to seismic data.

    Parameters
    ----------
    data_in : numpy array
        Input data
    window : int
        window length in number of samples (not time)
    time_axis : string, optional
        Confirm whether the input data has the time axis on the vertical or
        horizontal axis

    Returns
    -------
    y : Data with AGC applied

    tosum : AGC scaling values that were applied

    """
    if time_axis != 'vertical':
        data_in = data_in.T

    nt = data_in.shape[0]
    nx = data_in.shape[1]

    y = np.zeros((nt, nx))
    tosum = np.zeros((nt, nx))

    # enforce that window is integer
    if type(window) != int:
        window = int(window)

    # enforce that window is smaller than length of time series
    if window > nt:
        window = nt

    # enforce that window is odd
    if window % 2 == 0:
        window = window - 1

    len2 = int((window-1)/2)

    e = np.cumsum(abs(data_in), axis=0)

    for i in range(nt):
        if i-len2-1 > 0 and i+len2 < nt:
            tosum[i, :] = e[i+len2, :] - e[i-len2-1, :]
        elif i-len2-1 < 1:
            tosum[i, :] = e[i+len2, :]
        else:
            tosum[i, :] = e[-1, :] - e[i-len2-1, :]

    for i in range(len2):
        tosum[i, :] = tosum[len2+1, :]
        tosum[-1-i, :] = tosum[-1-len2, :]

    y = data_in / tosum

    if normalize:
        y = y/np.max(abs(y))

    return y, tosum
