import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import hann

import par_file as pf

# Set Times New Roman as the default font
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})


def plotChRange(data, time_index, startCh=0, endCh=None, fs=1000, clipPerc=None,
                time_axis='vertical', cmap='seismic', dpi=200, title=None,
                outfile=None):
    """
    Plot DAS profile for given channel range

    Parameters
    ----------
    data : numpy.ndarray
        Input data in the t-x domain
    startCh : int
        First channel considered, in case only part of the channels is
        of interest. Defaults to 0
    endCh: int
        Last channel considered,  in case only part of the channels is
        of interest.
    fs : int
        Sampling frequency in Hz
    dx : int
        Channel offset in m
    shift : bool, optional
        If set to True, zero frequency is shifted to the center.

    Returns
    -------
    data_fk : f-k spectrum of input dataset
    f_axis : corresponding frequency axis
    k_axis : corresponding wavenumber axis
    """

    if time_axis != 'vertical':
        data = data.T

    if not endCh:
        endCh = data.shape[1]

    if clipPerc:
        clip = np.percentile(np.absolute(data), clipPerc)
        plt.figure(dpi=dpi)
        plt.imshow(data, aspect='auto', interpolation='none',
                   vmin=-clip, vmax=clip, cmap=cmap)#, extent=(startCh, endCh-1,init_t, fin_t))
    else:
        plt.figure(dpi=dpi)
        plt.imshow(data, aspect='auto', interpolation='none',
                   vmin=-abs(data).max(), vmax=abs(data).max(),
                   cmap=cmap)#, extent=(startCh, endCh-1, init_t, fin_t))
    plt.xlabel('Channel number')
    plt.xticks(np.linspace(0, 500, 6).tolist(), np.linspace(startCh, startCh + 500, 6).tolist())
    plt.ylabel('Time (s)')
    plt.yticks(np.linspace(0, 660, 4).tolist(), np.linspace(time_index/220, (time_index+660)/220, 4).tolist())

    plt.colorbar(label='Strain rate (10$^{-9}$ s$^{-1}$)')
    plt.tight_layout()
    if title:
        plt.title(title, fontsize=16)
    if outfile:
        plt.savefig(outfile, format='png')
        plt.close('all')
    return

def fk_spectra(data, fs=1000, dx=4, shift=True, time_axis='vertical',
               zero_padding=True):
    """
    Function that calculates frequency-wavenumber spectrum of input data.

    Parameters
    ----------
    data : numpy.ndarray
        Input data in the t-x domain
    fs : int
        Sampling frequency in Hz
    dx : int
        Channel offset in m
    shift : bool, optional
        If set to True, zero frequency is shifted to the center.
    zero_padding : bool, optional
        Zero-pad signal to next power of two before fft. Defaults to true

    Returns
    -------
    data_fk : f-k spectrum of input dataset
    f_axis : corresponding frequency axis
    k_axis : corresponding wavenumber axis
    """
    if time_axis != 'vertical':
        data = data.T

    if zero_padding:
        next2power_nt = np.ceil(np.log2(data.shape[0]))
        next2power_nx = np.ceil(np.log2(data.shape[1]))
        nTi = int(2**next2power_nt)
        nCh = int(2**next2power_nx)
    else:
        nTi, nCh = data.shape

    if shift:
        data_fk = np.fft.fftshift(np.fft.fft2(data, s=(nTi, nCh)))
        f_axis = np.fft.fftshift(np.fft.fftfreq(nTi, d=1/fs))
        k_axis = np.fft.fftshift(np.fft.fftfreq(nCh, d=dx))

    else:
        data_fk = np.fft.fft2(data, s=(nTi, nCh))
        f_axis = np.fft.fftfreq(nTi, d=1/fs)
        k_axis = np.fft.fftfreq(nCh, d=dx)

    return data_fk, f_axis, k_axis


def fkr_spectra(data, fs=1000, dx=4, shift=True, time_axis='vertical',
                zero_padding=True):
    """
    Function that calculates frequency-wavenumber spectrum of real input
    data.

    Taking advantage that input signal is real to only compute posivite
    frequencies.

    Parameters
    ----------
    data : numpy.ndarray
        Input data in the t-x domain
    fs : int
        Sampling frequency in Hz
    dx : int
        Channel offset in m
    shift : bool, optional
        If set to True, zero frequency is shifted to the center.
    zero_padding : bool, optional
        Zero-pad signal to next power of two before fft. Defaults to true

    Returns
    -------
    data_fk : f-k spectrum of input dataset
    f_axis : corresponding frequency axis
    k_axis : corresponding wavenumber axis
    """
    if time_axis != 'vertical':
        data = data.T

    if zero_padding:
        next2power_nt = np.ceil(np.log2(data.shape[0]))
        next2power_nx = np.ceil(np.log2(data.shape[1]))
        nTi = int(2**next2power_nt)
        nCh = int(2**next2power_nx)
    else:
        nTi, nCh = data.shape

    if shift:
        data_fk = np.fft.fftshift(np.fft.rfft2(data, s=(nTi, nCh),
                                               axes=(1, 0)),
                                  axes=1)
        f_axis = np.fft.rfftfreq(nTi, d=1/fs)
        k_axis = np.fft.fftshift(np.fft.fftfreq(nCh, d=dx))

    else:
        data_fk = np.fft.rfft2(data, s=(nTi, nCh), axes=(1, 0))
        f_axis = np.fft.rfftfreq(nTi, d=1/fs)
        k_axis = np.fft.fftfreq(nCh, d=dx)

    return data_fk, f_axis, k_axis


def plotFkSpectra(data_fk, f_axis, k_axis, log=False, vmax=None, dpi=200,
                  title=None, outfile=None):
    """
    Plot fk-spectra of data

    Parameters
    ----------
    data_fk : numpy.ndarray
        f-k spectrum of input dataset
    f_axis : numpy.ndarray
        corresponding frequency axis
    k_axis : numpy.ndarray
        corresponding wavenumber axis
    log : bool, optional
        If set True, amplitude of plot is logarithmic
    vmax : float, optional
        Set max value of colormap. If set to None, colormap is applied to min
        and max value of the data
    dpi : int, optional
        Resolution of figure
    title : str, optional
        Title of the figure. If set to None, no title is plotted.
    outfile : str, optional
        Path where to save figure. If set to None, figure is not saved.

    Returns
    -------
    Plot of f-k spectra
    """
    extent = (k_axis[0], k_axis[-1], f_axis[-1], f_axis[0])
    plt.figure(dpi=dpi)
    if log:
        plt.imshow(np.log10(abs(data_fk)/abs(data_fk).max()),
                   aspect='auto', interpolation='none', extent=extent,
                   cmap='viridis', vmax=vmax)
    else:
        plt.imshow(abs(data_fk), aspect='auto', interpolation='none',
                   extent=extent, cmap='viridis', vmax=vmax)
    if title:
        plt.title(title)
    plt.xlabel('Wavenumber (1/m)')
    plt.ylabel('Frequency (1/s)')
    cbar = plt.colorbar()
    if log:
        cbar.set_label('Normalized Power [dB]')
    else:
        cbar.set_label('PSD')
    if outfile:
        plt.savefig(outfile)
    plt.show()
    return


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
