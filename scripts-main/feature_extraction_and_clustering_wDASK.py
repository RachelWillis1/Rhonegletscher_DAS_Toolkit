#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:48:27 2024

@author: rachelwillis
"""

import time
import joblib
import numpy as np
import pandas as pd
from scipy.signal.windows import hann
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import dask
import dask.delayed

import par_file as pf
# import plotting_functions as plot
from readers import das_reader as reader
import preprocessing_functions as pp


# ============================
#        INITIAL SETUP
# ============================

overall_start_time = time.time()  # Start overall execution timer


print(f"{pf.nr_files} .hdf5 files are in list to be processed.")

# Set Matplotlib Font
plt.rcParams["font.family"] = pf.plot_font_family
plt.rcParams.update({'font.size': pf.plot_font_size})

# ============================
#      COVARIANCE MATRIX
# ============================

def compute_features_averaging_window_local(startChLoc, noise_filt_agc, t_window_short=120,
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
    coh_matrix_ave = np.zeros((pf.nfreqs, pf.nstations, pf.nstations),
                              dtype=np.complex128)

    # Loop over each channel and compute STFT
    for subw_ind, subw_start in enumerate(range(0, pf.len_ave_window-pf.noverlap_sub, pf.noverlap_sub)):
        if len(st_ave_window[subw_start:subw_start+pf.len_sub_window]) == len(taper_window.reshape(-1, 1)):
            sub_window_tapered = st_ave_window[subw_start:subw_start+pf.len_sub_window] * taper_window.reshape(-1, 1)
            win_spectra = np.fft.rfft(sub_window_tapered, axis=0)

            mean_spectra = np.mean(win_spectra, axis=0)
            win_spectra -= mean_spectra

            normalizer = np.sum(np.absolute(win_spectra) ** 2, axis=0)
            normalizer = np.tile(normalizer, (normalizer.shape[0], 1))
            normalizer = normalizer * normalizer.T

            # Compute covariance matrix each time
            for freq in range(pf.nfreqs):
                welch_numerator = np.outer(win_spectra[freq, :],
                                      np.conj(win_spectra[freq, :]))
                welch_numerator = np.absolute(welch_numerator) ** 2
                coherence = np.multiply(welch_numerator, 1 / normalizer)

                coh_matrix_ave[freq, :, :] += coherence

    # Compute features from eigenvalue distribution
    eigenvalues_fk = np.zeros(pf.nfreqs)
    coherence_fk = np.zeros(pf.nfreqs)
    variance_fk = np.zeros(pf.nfreqs)

    for m in range(pf.nfreqs):
        eigenvals, eigenvec = np.linalg.eigh(coh_matrix_ave[m, :, :])
        eigenvals = np.sort(eigenvals)[::-1]
        eigenvals_ls.append(eigenvals)
        mean_eigenval =  np.sum(eigenvals)/len(eigenvals)

        # Extract features
        eigenvalues_fk[m] = eigenvals[0]
        coherence_fk[m] = eigenvals[0] / np.sum(eigenvals)
        variance_fk[m] = np.sum((eigenvals - mean_eigenval)**2) / len(eigenvals)

    features = np.stack([eigenvalues_fk, coherence_fk, variance_fk])
    
    # plot.plot_covariance_and_features(coh_matrix_ave=coh_matrix_ave, features=features,
    # startChLoc=startChLoc, starttime=starttime, time_index=time_index, save=True)


    return features.T, startChLoc, coh_matrix_ave


# ============================
#        MAIN EXECUTION
# ============================

if __name__ == '__main__':
    eigenvals_ls = []
    nr_space_loops = len(range(pf.startCh+50, pf.endCh-50-100, 500))
    catalogue_df = pd.DataFrame(columns=pf.cols)
    coh_features_ls = []
    startChLoc_ls = []
    time_index_ls = []
    filename_ls = []

    overall_preproc_time = 0
    overall_coh_time = 0
    overall_rf_time = 0

    for file_id in range(0, pf.nr_files - pf.nr_files_to_load, pf.nr_files_to_process):
        preproc_start_time = time.time()
        print(f'Processing File ID: {file_id}')

        # Load Data
        st = reader(pf.files_list[file_id:file_id + pf.nr_files_to_load], stream=False,
                    channels=pf.channels_to_read_new[pf.startCh:pf.endCh],
                    h5type='idas2', debug=True)

        if st[0].shape[0] != 90000:
            continue

        sampling_rate = int(st[2]['sampling_frequency_Hz'])
        starttime = UTCDateTime(st[2]['starttime'])

        # Preprocess Data
        time_mean = st[0].mean(axis=1).reshape(-1, 1)
        st_demean = st[0] - time_mean

        delayed_results = [dask.delayed(pp.preproc_test)(tr_id, st_demean) for tr_id in range(pf.nrTr)]
        results = dask.compute(*delayed_results)
        results_array = np.stack(results).T
        st_whitened = results_array[pf.start_sample:pf.end_sample, :]
        st_whitened = st_whitened - st_whitened.mean(axis=1).reshape(-1,1)

        # Apply AGC and Spatial Highpass
        st_whitened_agc, _ = pp.AGC(st_whitened, 400)
        delayed_results1 = [dask.delayed(pp.preproc_x)(tr_t_id, st_whitened_agc) for tr_t_id in range(st_whitened_agc.shape[0])]
        results1 = dask.compute(*delayed_results1)
        results_array1 = np.stack(results1)
        st_clean = results_array1[:, 50:-50]

        preproc_end_time = time.time()
        preproc_time = preproc_end_time - preproc_start_time
        overall_preproc_time += preproc_time
        # print(f"Preprocessing Time: {preproc_time:.2f} seconds")

        coh_start_time = time.time()
        # Features Extraction
        for i, time_index in enumerate(range(0, st_clean.shape[0], 660)):
            noise_filt_agc = st_clean[time_index:time_index + 660, :]

            # Extract Coherency Features
            delayed_coh_results = [dask.delayed(compute_features_averaging_window_local)(startCh, noise_filt_agc) for startCh in range(0, pf.nrTr - 50 - 50, 500)]
            coh_results = dask.compute(*delayed_coh_results)

            for i in range(len(coh_results)):
                features, startChLoc, coh_matrix_ave = coh_results[i]  # Unpack the tuple
                coh_features_ls.append(features)  # Append features
                startChLoc_ls.append(startChLoc)
                time_index_ls.append(time_index/660 * 3)
                filename_ls.append(pf.files_list[file_id])

                # plot.plotChRange(noise_filt_agc[:, startChLoc:startChLoc+500], time_index, startCh=startChLoc, endCh=startChLoc + 500, fs=250,
                #                           clipPerc=99.97, title = f'{starttime}.png',
                #                           outfile=f'{starttime}_ch_{startChLoc}_to_{startChLoc+500}_tind_{time_index/220}_to_{(time_index+660)/220}.png')

        coh_end_time = time.time()
        coh_time = coh_end_time - coh_start_time
        overall_coh_time += coh_time
        # print(f"Covariance Matrix Computation Time: {coh_time:.2f} seconds")


    # Ensure the lists are numpy arrays and reshape them for concatenation
    startChLoc_array = np.array(startChLoc_ls).reshape(-1, 1)  # Column vector
    time_index_array = np.array(time_index_ls).reshape(-1, 1)  # Column vector
    filename_array = np.array(filename_ls).reshape(-1, 1)  # Column vector (filenames as strings)

    # Combine the lists into a single array (columns)
    metadata_array = np.hstack([startChLoc_array, time_index_array, filename_array])

    coh_features_array = np.stack(coh_features_ls)
    coh_features_reshaped = np.reshape(coh_features_array, (coh_features_array.shape[0], coh_features_array.shape[1] * coh_features_array.shape[2])) #[S1T1, S1T2,..., S2T1,...]

    # Ensure `coh_features_reshaped` has the same number of rows as the metadata
    if metadata_array.shape[0] != coh_features_reshaped.shape[0]:
        raise ValueError("Mismatch in the number of rows between metadata and features.")

    combined_array = np.hstack([metadata_array, coh_features_reshaped])

    # Save all calculated features to disc
    np.save(f'{pf.output_dir}meta_data_coherency.npy', combined_array)
    # np.save('meta_data_coherency.npy', combined_array)

    # Load trained model and predict clusters
    rf_start_time = time.time()
    rf_model = joblib.load(pf.ml_model)
    rf_pred = rf_model.predict(combined_array[:, 3:])
    rf_end_time = time.time()

    rf_time = rf_end_time - rf_start_time
    overall_rf_time += rf_time
    # print(f"Clustering Computation Time: {rf_time:.2f} seconds")

    np.save(f'{pf.output_dir}rf_pred.npy', rf_pred)
    # np.save('rf_pred.npy', rf_pred)


# End the timer
overall_end_time = time.time()
print(f"Overall Preprocessing Time: {overall_preproc_time:.2f} seconds")
print(f"Overall Covariance Matrix Computation Time: {overall_coh_time:.2f} seconds")
print(f"Overall Clustering Time: {overall_rf_time:.2f} seconds")
print(f"Overall Execution Time: {overall_end_time - overall_start_time:.2f} seconds")
