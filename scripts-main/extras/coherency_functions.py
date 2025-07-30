import numpy as np
from scipy.signal import stft, hann


def stream2array(stream, startCh=0, endCh=None):
    """
    Convert Stream Object to 2d numpy array

    Parameters
    ----------
    stream : obspy.core.stream.Stream
        Stream object containing seismic traces
    startCh : int
        First channel considered, in case only part of the channels is
        of interest. Defaults to 0
    endCh: int
        Last channel considered

    Returns
    -------
    ensemble : 2d numpy array of input stream
    """
    if not endCh:
        endCh = len(stream)

    nStations = endCh - startCh
    nTimePoints = stream[startCh].stats.npts
    ensemble = np.zeros((nTimePoints, nStations))

    for channelNumber in range(startCh, endCh):
        ensemble[:, channelNumber-startCh] = stream[channelNumber].data

    return ensemble


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


def time_taper_1D(taper_length, max_perc=0.05):
    taper_multiplier = np.ones(taper_length)
    # If no edge_length is given, taper 50% of input data
    edge_length = int(max_perc*taper_length)
    # Make sure edge_length is odd
    if edge_length % 2 == 0:
        edge_length += 1
    # Create hanning window, split it and set values to 1 inbetween
    taper_window = hann(2*edge_length)
    taper_multiplier[:edge_length] = taper_window[:edge_length]
    taper_multiplier[-edge_length:] = taper_window[-edge_length:]
    return taper_multiplier


def taper_1D(data, edge_length=None):
    taper_multiplier = np.ones(len(data))
    # If no edge_length is given, taper 50% of input data
    if not edge_length:
        edge_length = len(data) // 4
    # Make sure edge_length is odd
    if edge_length % 2 == 0:
        edge_length += 1
    # Create hanning window, split it and set values to 1 inbetween
    taper_window = hann(2*edge_length)
    taper_multiplier[:edge_length] = taper_window[:edge_length]
    taper_multiplier[-edge_length:] = taper_window[-edge_length:]
    # Multiply each trace with the constructed taper window
    data_tapered = data * taper_multiplier

    return data_tapered


def double_whiten(data_in, freq_smooth=10, t_smooth=100):
    """
    Function that applies first spectral whitening and afterwards
    temporal normalisation.

    Parameters
    ----------
    data_in : 1D numpy array
        Input data
    freq_smooth : int
        window length in number of samples for smoothing in
        frequency domain
    t_smooth : int
        window length in number of samples for smoothing trace in
        time domain

    Returns
    -------
    double_norm : Spectral whitened and time normalized trace
    """
    spectrum = np.fft.rfft(data_in)
    spectrum_norm = AGC_1D(spectrum, freq_smooth)
    whitened = np.fft.irfft(spectrum_norm)
    double_norm = AGC_1D(whitened, t_smooth)
    return double_norm


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


def compute_cov_matrix_averaging_window(st, time_index, t_window_long=10000,
                                        t_window_short=300, fs=1000):
    """
    Function to compute covariance matrix for a single averaging window.

    Parameters
    ----------
    st : Obspy Stream
        Input data in obspy stream object
    time_index : int
        Current time window to be considered
    t_window_long : int
        Length of averaging window
    t_window_short : int
        Length of sub-window
    fs : int
        Sampling frequency in Herz

    Returns
    -------
    cov_matrix_ave / nStations : Covariance matrix for input time window
    """
    # Pre-allocate covariance matrix for averaging window
    cov_matrix_ave = np.zeros(
        (nFreqs, nStations, nStations), dtype=np.complex128)
    # Pre-allocate data vector for sub-window
    data_vector_sub = np.zeros(
        (nStations, nFreqs, nr_sub_win), dtype=np.complex128)

    # Loop over each channel and compute STFT
    for i, nTr in enumerate(array_stations):
        tr = st[nTr].data[
            time_index*t_window_long:(time_index+1)*t_window_long
            ]
        _, _, Zxx = stft(tr, fs, nperseg=t_window_short)
        data_vector_sub[i, :, :] = Zxx

    # Loop over all sub-windows and frequencies and compute covariance matrix
    # each time
    for sub_win in range(nr_sub_win):
        for freq in range(nFreqs):
            cov_matrix_sub = np.outer(data_vector_sub[:, freq, sub_win],
                                      np.conj(data_vector_sub[
                                          :, freq, sub_win]))
            cov_matrix_ave[freq, :, :] += cov_matrix_sub

    return cov_matrix_ave / nStations


def compute_cov_matrix_averaging_window2(st_ave_window, t_window_short=120,
                                         fs=200, taper_perc=0.25,
                                         nstations=67):
    """
    Function to compute covariance matrix for a single averaging window.

    Parameters
    ----------
    st_ave_window : 1D numpy array
        Input window for which covariance matrix is computed
    t_window_short : int
        Length of sub-window
    t_window_long : int
        Length of averaging window
    fs : int
        Sampling frequency in Herz
    taper_perc : float
        Proportion of window to be tapered

    Returns
    -------
    cov_matrix_ave : Covariance matrix for input time window
    """
    # Compute taper-window for sub-windows
    nfreqs = t_window_short // 2 + 1
    taper_window = np.ones(t_window_short)
    side_len = int(taper_perc * t_window_short)
    taper_sides = hann(2*side_len)
    taper_window[0:side_len] = taper_sides[0:side_len]
    taper_window[-side_len:] = taper_sides[-side_len:]

    # Pre-allocate covariance matrix for averaging window
    cov_matrix_ave = np.zeros(
        (nfreqs, nstations, nstations), dtype=np.complex128)

    # Pre-allocate complex spectras for all sub windows
    data_vector = np.zeros((nfreqs, nstations), dtype=np.complex128)

    # Loop over each channel and compute STFT
    for subw_ind, subw_start in enumerate(range(
            0, len_ave_window-noverlap_sub, noverlap_sub)):
        sub_window_tapered = st_ave_window[
            subw_start:subw_start+len_sub_window] * taper_window.reshape(-1, 1)
        data_vector = np.fft.rfft(sub_window_tapered, axis=0)

        # Compute covariance matrix each time
        for freq in range(nfreqs):
            cov_matrix_sub = np.outer(data_vector[freq, :],
                                      np.conj(data_vector[freq, :]))
            cov_matrix_ave[freq, :, :] += cov_matrix_sub

    return cov_matrix_ave / nr_sub_windows


def slant_stack_old(data_tx, velocities, dx, fs):
    """
    Perform a slant-stack on the given wavefield data.
    Parameters
    ----------
    array : ndarray
        Two-dimensional array object.
    velocities : ndarray
        One-dimensional array of trial velocities.
    fs : int
        Sampling frequency in Hz
    dx : int
        Channel offset in m
    Returns
    -------
    tuple
        Of the form `(tau, slant_stack)` where `tau` is an ndarray
        of the attempted intercept times and `slant_stack` are the
        slant-stacked waveforms.
    """
    dt = 1 / fs
    npts = data_tx.shape[1]
    nchannels = data_tx.shape[0]
    position = np.linspace(0, (nchannels)*dx, nchannels, endpoint=False)

    diff = position[1:] - position[:-1]
    diff = diff.reshape((len(diff), 1))
    ntaus = npts - int(np.max(position)*np.max(1/velocities)/dt) - 1
    slant_stack = np.empty((len(velocities), ntaus))
    rows = np.tile(np.arange(nchannels).reshape(nchannels, 1), (1, ntaus))
    cols = np.tile(np.arange(ntaus).reshape(1, ntaus), (nchannels, 1))

    pre_float_indices = position.reshape(nchannels, 1)/dt
    previous_lower_indices = np.zeros((nchannels, 1), dtype=int)
    for i, velocity in enumerate(velocities):
        float_indices = pre_float_indices/velocity
        lower_indices = np.array(float_indices, dtype=int)
        delta = float_indices - lower_indices
        cols += lower_indices - previous_lower_indices
        amplitudes = data_tx[rows, cols] * \
            (1-delta) + data_tx[rows, cols+1]*delta
        integral = 0.5*diff*(amplitudes[1:, :] + amplitudes[:-1, :])
        summation = np.sum(integral, axis=0)
        slant_stack[i, :] = summation

        previous_lower_indices[:] = lower_indices
    taus = np.arange(ntaus)*dt
    return (taus, slant_stack)


def slant_stack_full(data_tx, velocities, dx, fs):
    """
    Perform a slant-stack on the given wavefield data. Ensures
    that tau and t axis are equal length by expanding input
    array with zeros.

    Parameters
    ----------
    array : ndarray
        Two-dimensional array object.
    velocities : ndarray
        One-dimensional array of trial velocities.
    fs : int
        Sampling frequency in Hz
    dx : int
        Channel offset in m
    Returns
    -------
    tuple
        Of the form `(tau, slant_stack)` where `tau` is an ndarray
        of the attempted intercept times and `slant_stack` are the
        slant-stacked waveforms.
    """
    dt = 1 / fs
    npts = data_tx.shape[1]
    nchannels = data_tx.shape[0]
    position = np.linspace(0, (nchannels)*dx, nchannels, endpoint=False)

    diff = position[1:] - position[:-1]
    diff = diff.reshape((len(diff), 1))
    ntaus = npts - int(np.max(position)*np.max(1/velocities)/dt) - 1

    data_tx = np.concatenate((data_tx, np.zeros((nx, npts-ntaus))), axis=1)
    ntaus = npts

    slant_stack = np.empty((len(velocities), ntaus))
    rows = np.tile(np.arange(nchannels).reshape(nchannels, 1), (1, ntaus))
    cols = np.tile(np.arange(ntaus).reshape(1, ntaus), (nchannels, 1))

    pre_float_indices = position.reshape(nchannels, 1)/dt
    previous_lower_indices = np.zeros((nchannels, 1), dtype=int)
    for i, velocity in enumerate(velocities):
        float_indices = pre_float_indices/velocity
        lower_indices = np.array(float_indices, dtype=int)
        delta = float_indices - lower_indices
        cols += lower_indices - previous_lower_indices
        amplitudes = data_tx[rows, cols] * \
            (1-delta) + data_tx[rows, cols+1]*delta
        integral = 0.5*diff*(amplitudes[1:, :] + amplitudes[:-1, :])
        summation = np.sum(integral, axis=0)
        slant_stack[i, :] = summation

        previous_lower_indices[:] = lower_indices
    taus = np.arange(ntaus)*dt
    return (taus, slant_stack)


def slant_stack(tmatrix, velocities, dx, dt):
    """Perform a slant-stack on the given wavefield data.
    Parameters
    ----------
    array : Array1d
        One-dimensional array object.
    velocities : ndarray
        One-dimensional array of trial velocities.
    Returns
    -------
    tuple
        Of the form `(tau, slant_stack)` where `tau` is an ndarray
        of the attempted intercept times and `slant_stack` are the
        slant-stacked waveforms.
    """
    npts = tmatrix.shape[1]
    nchannels = tmatrix.shape[0]
    position = np.linspace(0, (nchannels)*dx, nchannels, endpoint=False)

    diff = position[1:] - position[:-1]
    diff = diff.reshape((len(diff), 1))
    ntaus = npts - int(np.max(position)*np.max(1/velocities)/dt) - 1

    tmatrix = np.concatenate(
        (tmatrix, np.zeros((nchannels, npts-ntaus))), axis=1)
    ntaus = npts

    slant_stack = np.empty((len(velocities), ntaus))
    rows = np.tile(np.arange(nchannels).reshape(nchannels, 1), (1, ntaus))
    cols = np.tile(np.arange(ntaus).reshape(1, ntaus), (nchannels, 1))

    pre_float_indices = position.reshape(nchannels, 1)/dt
    previous_lower_indices = np.zeros((nchannels, 1), dtype=int)
    for i, velocity in enumerate(velocities):
        float_indices = pre_float_indices/velocity
        lower_indices = np.array(float_indices, dtype=int)
        delta = float_indices - lower_indices
        cols += lower_indices - previous_lower_indices
        amplitudes = tmatrix[rows, cols] * \
            (1-delta) + tmatrix[rows, cols+1]*delta
        integral = 0.5*diff*(amplitudes[1:, :] + amplitudes[:-1, :])
        summation = np.sum(integral, axis=0)
        slant_stack[i, :] = summation

        previous_lower_indices[:] = lower_indices
    return slant_stack


def extract_velocities(data_in, pmin=0, pmax=1/800):
    """Extract apparent velocities using tau-p transform."""
    p = np.linspace(pmin, pmax, 100)
    velocities = 1 / (p + 1e-8)
    tau_pi = slant_stack_full(data_in[:, ::1].T, velocities, dx=4, dt=1/fs)
    f_p_real = np.abs(np.fft.rfft(tau_pi, axis=-1))
    f_p_real_power = f_p_real**2
    velo_energy = f_p_real_power[:, 150:450].sum(axis=1)
    feature1 = decimate(velo_energy, 5)
    velo_energy = f_p_real_power[:, 450:750].sum(axis=1)
    feature2 = decimate(velo_energy, 5)

    # Flip array to get negative velocities
    tau_pi = slant_stack_full(data_in[:, ::-1].T, velocities, dx=4, dt=1/fs)
    f_p_real = np.abs(np.fft.rfft(tau_pi, axis=-1))
    f_p_real_power = f_p_real**2
    velo_energy = f_p_real_power[:, 150:450].sum(axis=1)
    feature3 = decimate(velo_energy, 5)
    velo_energy = f_p_real_power[:, 450:750].sum(axis=1)
    feature4 = decimate(velo_energy, 5)
    scaler_low = feature1.sum() + feature3.sum()
    scaler_high = feature2.sum() + feature4.sum()

    return (feature1/scaler_low, feature2/scaler_high,
            feature3/scaler_low, feature4/scaler_high)


def energy_norm(data_tx):
    tosum = (data_tx**2).sum(axis=0)
    return data_tx / tosum.reshape(1, -1)


def Gauss_smooth_freq(f_axis, f_cutoff, Gsmooth=0.2):
    f_diff = f_axis-f_cutoff
    return np.exp(-(f_diff/f_cutoff)**2/Gsmooth**2)


def freq_taper_1D(f_axis, f_cutoff_low=5, f_cutoff_high=80,
                  Gsmooth_flow=0.2, Gsmooth_fhigh=0.2):
    f_diff_max = f_axis-f_cutoff_high
    f_diff_min = f_axis-f_cutoff_low

    freq_taper = np.ones(len(f_axis))
    gaussian_max = Gauss_smooth_freq(f_axis, f_cutoff=f_cutoff_high,
                                     Gsmooth=Gsmooth_fhigh)
    gaussian_min = Gauss_smooth_freq(f_axis, f_cutoff=f_cutoff_low,
                                     Gsmooth=Gsmooth_flow)

    freq_taper[f_diff_max > 0] = gaussian_max[f_diff_max > 0]
    freq_taper[f_diff_min < 0] = gaussian_min[f_diff_min < 0]

    return freq_taper


def compute_cov_matrix_averaging_window_full(
        st_ave_window, nfreqs, nstations, len_ave_window, noverlap_sub,
        len_sub_window, nr_sub_windows, t_window_short=120, fs=200,
        taper_perc=0.25):
    """
    Function to compute covariance matrix for a single averaging window.

    Parameters
    ----------
    st_ave_window : 1D numpy array
        Input window for which covariance matrix is computed
    t_window_short : int
        Length of sub-window
    t_window_long : int
        Length of averaging window
    fs : int
        Sampling frequency in Herz
    taper_perc : float
        Proportion of window to be tapered

    Returns
    -------
    cov_matrix_ave : Covariance matrix for input time window
    """
    # Compute taper-window for sub-windows
    taper_window = np.ones(t_window_short)
    side_len = int(taper_perc * t_window_short)
    taper_sides = hann(2*side_len)
    taper_window[0:side_len] = taper_sides[0:side_len]
    taper_window[-side_len:] = taper_sides[-side_len:]

    # Pre-allocate covariance matrix for averaging window
    cov_matrix_ave = np.zeros(
        (nfreqs, nstations, nstations), dtype=np.complex128)

    # Pre-allocate complex spectras for all sub windows
    data_vector = np.zeros((nfreqs, nstations), dtype=np.complex128)

    # Loop over each channel and compute STFT
    for subw_ind, subw_start in enumerate(range(0, len_ave_window-noverlap_sub,
                                                noverlap_sub)):
        sub_window_tapered = (st_ave_window[subw_start:subw_start
                                            + len_sub_window]
                              * taper_window.reshape(-1, 1))
        data_vector = np.fft.rfft(sub_window_tapered, axis=0)

        # Compute covariance matrix each time
        for freq in range(nfreqs):
            cov_matrix_sub = np.outer(data_vector[freq, :],
                                      np.conj(data_vector[freq, :]))
            cov_matrix_ave[freq, :, :] += cov_matrix_sub

    return cov_matrix_ave / nr_sub_windows
