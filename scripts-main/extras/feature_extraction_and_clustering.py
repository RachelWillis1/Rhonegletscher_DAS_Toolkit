#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:01:18 2024

@author: rachelwillis
"""

from scipy.signal import iirfilter, zpk2sos, sosfilt, hann, sosfreqz, decimate
import json
import multiprocessing
import numpy as np
import pandas as pd
import glob
import time
from obspy import UTCDateTime, Stream, Trace
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import dask.array as da
import dask

from pydvs import coherency_functions
from pydvs import fk_functions
from readers import das_reader as reader
from par_file import *

# Start the timer
start_time = time.time()

npts = int(nr_files_to_load * fs_orig * 30)
files_list = sorted(glob.glob(data_directory + "*.hdf5"))
nr_files = len(files_list)
print(f"{nr_files} .hdf5 files are in list to be processed.")

# Start and end sample (to avoid tapered data)
start_sample = int(30*fs)
end_sample = int((nr_files_to_load-1)*30*fs+(len_ave_window-noverlap_ave))

nstations = len(channels_example)
nfreqs = len_sub_window // 2 + 1

# Number of time windows
n_twin = (((nr_files-nr_files_to_load) // nr_files_to_process)
          * int(nr_files_to_process * 30 * fs / noverlap_ave))

# Define taper
wlen = int(max_perc * npts)
taper_sides = hann(2*wlen+1)
taper = np.hstack((taper_sides[:wlen], np.ones(npts-2*wlen),
                   taper_sides[len(taper_sides)-wlen:]))

fe = 0.5 * fs_orig
low = freqmin / fe
high = freqmax / fe
z, p, k = iirfilter(corners, [low, high], btype='band', ftype='butter',
                    output='zpk')
sos = zpk2sos(z, p, k)

fe2 = 0.5 * fs
low2 = freqmin / fe2
high2 = freqmax / fe2
z2, p2, k2 = iirfilter(corners, [low2, high2],
                       btype='band', ftype='butter', output='zpk')
# f_axis_s = np.fft.rfftfreq(len_sub_window, d=1/fs)
f_axis_l = np.fft.rfftfreq(int(nr_files_to_load*30*fs), d=1/fs) # frequency domain taper
sos2 = zpk2sos(z2, p2, k2)
w, h = sosfreqz(sos2, len(f_axis_l))
f_taper = np.abs(h)  # Filter in frequency domain

# Design filter for spatial highpass
wlen_x = int(max_perc * nrTr)
taper_sides_x = hann(2*wlen_x+1)
taper_x = np.hstack((taper_sides_x[:wlen_x], np.ones(nrTr-2*wlen_x),
                     taper_sides_x[len(taper_sides_x)-wlen_x:]))

# Define highpass filter parameters
fe_x = 0.5 * df_x
f = flow / fe_x
zx, px, kx = iirfilter(corners_x, f, btype='highpass', ftype='butter',
                       output='zpk')
sos_x = zpk2sos(zx, px, kx)

#
# Define functions for preprocessing and extracting features
#


def preproc(tr_id, dec_factor=5):
    """Taper, filter and decimate along time-axis."""
    npts = len(st_demean[:, tr_id])
    wlen = int(max_perc * npts)
    taper_sides = hann(2*wlen+1)
    taper = np.hstack((taper_sides[:wlen], np.ones(npts-2*wlen),
                   taper_sides[len(taper_sides)-wlen:]))
    tapered = st_demean[:, tr_id] * taper
    firstpass = sosfilt(sos, tapered)
    filtered = sosfilt(sos, firstpass[::-1])[::-1]
    downsampled = filtered[::dec_factor]
    filtered_white = coherency_functions.single_whiten_taper(downsampled, f_taper=f_taper,
                                          f_smooth=10)
    return filtered_white


def preproc_x(tr_t_id):
    """Taper, filter and decimate along space-axis."""
    tapered = st_whitened_agc[tr_t_id, :] * taper_x
    firstpass = sosfilt(sos_x, tapered)
    filtered = sosfilt(sos_x, firstpass[::-1])[::-1]
    return filtered


def slant_stack_full(tmatrix, velocities, dx, dt):
    """Perform a slant-stack on the given wavefield data.

    The following function code has been modified from the distpy module.
    https://github.com/Schlumberger/distpy/tree/master/distpy
    Access: April 5, 2021

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

    tmatrix = np.concatenate((tmatrix, np.zeros((nchannels, npts-ntaus))),
                             axis=1)
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
    taus = np.arange(ntaus)*dt
    return (taus, slant_stack)


def extract_velocities(stCh, pmin=1/10000, pmax=1/1000):
    """Extract apparent velocities using tau-p transform."""
    data_in = noise_filt_agc[:, stCh:stCh+500]#+200]
    p = np.linspace(pmin, pmax, 100)
    velocities = 1 / (p + 1e-8)
    taus, tau_pi = slant_stack_full(data_in[:, ::1].T, velocities,
                                    dx=4, dt=1/fs)
    f_p_real = np.abs(np.fft.rfft(tau_pi, axis=-1))
    velo_energy = f_p_real[:, 50:100].sum(axis=1) #150:450].sum(axis=1)
    feature1 = decimate(velo_energy, 5)
    velo_energy = f_p_real[:, 100:150].sum(axis=1) #450:750].sum(axis=1)
    feature2 = decimate(velo_energy, 5)

    # Flip array to get negative velocities
    taus, tau_pi = slant_stack_full(data_in[:, ::-1].T, velocities,
                                    dx=4, dt=1/fs)
    f_p_real = np.abs(np.fft.rfft(tau_pi, axis=-1))
    velo_energy = f_p_real[:, 50:100].sum(axis=1) #150:450].sum(axis=1)
    feature3 = decimate(velo_energy, 5)
    velo_energy = f_p_real[:, 100:150].sum(axis=1) #450:750].sum(axis=1)
    feature4 = decimate(velo_energy, 5)
    
    scaler_low = feature1.sum() + feature3.sum()
    scaler_high = feature2.sum() + feature4.sum()


    return (feature1/scaler_low, feature2/scaler_high,
            feature3/scaler_low, feature4/scaler_high)

def compute_features_averaging_window_local(startChLoc, t_window_short=120,
                                            fs=200, taper_perc=0.25):
    """
    Function to compute covariance matrix for a single averaging window.

    Parameters
    ----------
    t_window_short : int
        Length of sub-window
    t_window_long : int
        Length of averaging window
    fs : int
        Sampling frequency in Hertz
    taper_perc : float
        Proportion of window to be tapered

    Returns
    -------
    cov_matrix_ave : Covariance matrix for input time window
    """
    channels = list(range(startChLoc, startChLoc+500, 4))
    st_ave_window = noise_filt_agc[:, channels]

    # Compute taper-window for sub-windows
    taper_window = np.ones(t_window_short)
    side_len = int(taper_perc * t_window_short)
    taper_sides = hann(2*side_len)
    taper_window[0:side_len] = taper_sides[0:side_len]
    taper_window[-side_len:] = taper_sides[-side_len:]

    # Pre-allocate covariance matrix for averaging window
    cov_matrix_ave = np.zeros((nfreqs, nstations, nstations),
                              dtype=np.complex128)

    # Pre-allocate complex spectras for all sub windows
    data_vector = np.zeros((nfreqs, nstations), dtype=np.complex128)

    # Loop over each channel and compute STFT
    for subw_ind, subw_start in enumerate(range(0, len_ave_window-noverlap_sub,
                                                noverlap_sub)):
        if len(st_ave_window[subw_start:subw_start+len_sub_window]) == len(taper_window.reshape(-1, 1)):
            sub_window_tapered = st_ave_window[subw_start:subw_start+len_sub_window] * taper_window.reshape(-1, 1)
            data_vector = np.fft.rfft(sub_window_tapered, axis=0)

            # Compute covariance matrix each time
            for freq in range(nfreqs):
                cov_matrix_sub = np.outer(data_vector[freq, :],
                                      np.conj(data_vector[freq, :]))
                cov_matrix_ave[freq, :, :] += cov_matrix_sub

    cov_matrix_ave /= nr_sub_windows

    # Compute features from eigenvalue distribution
    eigenvalues_fk = np.zeros(nfreqs)
    coherence_fk = np.zeros(nfreqs)
    variance_fk = np.zeros(nfreqs)
    # shannon_fk = np.zeros(nfreqs)

    for m in range(nfreqs):
        w, v = np.linalg.eigh(cov_matrix_ave[m, :, :])
        wlen = len(w)
        w_real = np.abs(w)
        indices = np.flip(np.argsort(w_real))
        w_real = w_real[indices]
        w1 = w_real[0]
        w_sum = sum(w_real)
        mean = w_sum/wlen
        # w_norm = w_real/w_sum
    
        # Extract features
        eigenvalues_fk[m] = w1
        coherence_fk[m] = w1 / w_sum
        variance_fk[m] = sum((w_real - mean)**2) / wlen
    #   shannon_fk[m] = sum(-w_norm * np.log(w_norm))
    
        features = np.stack([eigenvalues_fk, coherence_fk, variance_fk])
        
        
    # print('Plotting Event:{}'.format(file_id))
    # outfile = f"../output/event_pngs/ind_{file_id}_time_{time_index}_startCh_{startChLoc}.png"
    # fk_functions.plotChRange(st_ave_window, time_index, startCh=startChLoc, endCh=startChLoc+500, fs=fs,
    #                           clipPerc=99.97, title=f"{file_id}",outfile=outfile)

    return features.T


#
# Start looping over all files. Only look at upper channel segment (data
# quality is very low on lower channels since there is no snow.)
# Fk-filter implemented
#

if __name__ == '__main__':
    
    # fig, axs = plt.subplots(nrows=10, ncols=3, figsize=(13,11))

    nr_space_loops = len(range(startCh+50, endCh-50-100, 500)) # 100)) 
    velo_features = np.zeros((nr_space_loops, n_twin, 4, 20))
    coh_features = np.zeros((nr_space_loops, n_twin, nfreqs, 3))


    time_counter = 0
    for file_id in range(0, nr_files - nr_files_to_load,
                         nr_files_to_process):
        print(f'File ID: {file_id}')
        st = reader(files_list[file_id:file_id+nr_files_to_load], stream=False,
                    channels=channels_to_read_new[startCh:endCh],
                    h5type='idas2', debug=True)
        npts = st[0].shape[0]
        nrTr = st[0].shape[1]
        print('Format of data: ({}, {})'.format(npts, nrTr))
        print('')
        if npts != 90000:
            print("Length mismatch: skipping file.")
            continue
        # Remove time-mean from data
        time_mean = st[0].mean(axis=1).reshape(-1, 1)
        st_demean = st[0] - time_mean

        print("Starting parallelized pre-processing (filtering+whitening)...")
        # Create pool with X processors
        pool0 = multiprocessing.Pool(n_processes)
        results = pool0.map(preproc, range(nrTr))
        pool0.close()
        
        results_array = np.stack(results).T
        st_whitened = results_array[start_sample:end_sample, :]
        print('Complete! Quick check... Format of results: ({},{})'.format(
            st_whitened.shape[0], st_whitened.shape[1]))
        print('')
        st_whitened = st_whitened - st_whitened.mean(axis=1).reshape(-1,1) # was commented out
        st_whitened_agc, tosum = fk_functions.AGC(st_whitened, 400) # was 600

        print("Starting parallelized pre-processing (high-pass space)...")
        # Create pool with X processors
        pool1 = multiprocessing.Pool(n_processes)
        results1 = pool1.map(preproc_x, range(st_whitened_agc.shape[0]))
        pool1.close()
        results_array3 = np.stack(results1)
        st_clean = results_array3[:, 50:-50]
        print('Complete! Quick check... Format of results: ({},{})'.format(
            st_clean.shape[0], st_clean.shape[1]))

        print('Starting feature extraction...')
        for time_index in range(0, st_whitened.shape[0]-2*noverlap_ave,
                                len_ave_window-noverlap_ave):

            noise_filt_agc = st_clean[time_index:time_index+len_ave_window-noverlap_ave, :]

            ## . . Extract Velocity Features
            pool2 = multiprocessing.Pool(n_processes)
            velo_results = pool2.map(extract_velocities,
                                 range(0, nrTr-50-50, 500)) # nrTr-100-100, 100))
            pool2.close()
            velo_features[:, time_counter, :, :] = np.stack(velo_results)
            
            ## . . Extract Coherency Features
            pool3 = multiprocessing.Pool(n_processes)
            coh_results = pool3.map(compute_features_averaging_window_local,range(0, nrTr-50-50, 500))#1800, 450))
            pool3.close()
            coh_features[:, time_counter, :, :] = np.stack(coh_results)
            # Advance `time_counter` globally
            time_counter += 1
            
    velo_features_reshaped = np.reshape(velo_features, (velo_features.shape[0] * velo_features.shape[1], velo_features.shape[2] * velo_features.shape[3])) #[S1T1, S1T2,..., S2T1,...]
    mask = velo_features_reshaped[:,0] != 0    
    velo_features_reshaped = velo_features_reshaped[mask]
    
    velo_kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(velo_features_reshaped)
    velo_kmeans_ls = []
    for i in velo_kmeans.labels_:
        velo_kmeans_ls.append(i)
        
    velo_agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete').fit(velo_features_reshaped)
    velo_agg_ls = []
    for i in velo_agg.labels_:
        velo_agg_ls.append(i)
    
    # Save all calculated features to disc
    np.save('../output/extr_features/velo'+'_temp.npy', velo_features_reshaped)
    np.save('../output/velo_kmeans.npy', velo_kmeans_ls)
    np.save('../output/velo_agg.npy', velo_agg_ls)

    coh_features_reshaped = np.reshape(coh_features, (coh_features.shape[0] * coh_features.shape[1], coh_features.shape[2] * coh_features.shape[3])) #[S1T1, S1T2,..., S2T1,...]
    mask = coh_features_reshaped[:,0] != 0    
    coh_features_reshaped = coh_features_reshaped[mask]
    
    coh_kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coh_features_reshaped)
    coh_kmeans_ls = []
    for i in coh_kmeans.labels_:
        coh_kmeans_ls.append(i)
        
    coh_agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete').fit(coh_features_reshaped)
    coh_agg_ls = []
    for i in coh_agg.labels_:
        coh_agg_ls.append(i)
    
    # Save all calculated features to disc
    np.save('../output/extr_features/coh'+'_temp.npy', coh_features_reshaped)
    np.save('../output/extr_features/coh_kmeans.npy', coh_kmeans_ls)
    np.save('../output/extr_features/coh_agg.npy', coh_agg_ls)


# End the timer
end_time = time.time()

# Calculate the time difference
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")