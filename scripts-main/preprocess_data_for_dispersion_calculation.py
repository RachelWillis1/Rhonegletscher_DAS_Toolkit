#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:01:15 2025

@author: rachelwillis
"""

import time
import numpy as np
from scipy.signal import sosfilt, hann
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import dask
import dask.delayed

import par_file as pf
from readers import das_reader as reader
import preprocessing_functions as pp

import coherency_functions


# ============================
#        INITIAL SETUP
# ============================

# Start Timer
start_time = time.time()

print(f"{pf.nr_files} .hdf5 files are in list to be processed.")

# Set Matplotlib Font
plt.rcParams["font.family"] = pf.plot_font_family
plt.rcParams.update({'font.size': pf.plot_font_size})


# ============================
#        MAIN EXECUTION
# ============================

if __name__ == '__main__':
    
    all_st_clean = []
    all_starttimes = []


    for file_id in range(0, pf.nr_files - pf.nr_files_to_load, pf.nr_files_to_process):
        print(f'Processing File ID: {file_id}')

        # Load Data
        st = reader(pf.files_list[file_id:file_id + pf.nr_files_to_load], stream=False,
                    channels=pf.channels_to_read_new[pf.startCh:pf.endCh],
                    h5type='idas2', debug=True)
        
        if st[0].shape[0] != 90000:
            continue

        sampling_rate = int(st[2]['sampling_frequency_Hz'])
        starttime = UTCDateTime(st[2]['starttime'])
        
        # Check if the file's starttime is within the filtering window
        if starttime < pf.target_starttime or starttime > pf.target_endtime:
            print(f"Skipping File {file_id}: Outside time range")
            continue  # Skip processing this file

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
        
        # Append the processed st_clean to the list
        all_st_clean.append(st_clean)
        all_starttimes.append(str(starttime))  # Convert starttime to string for saving

    # Convert lists to a NumPy format
    all_st_clean_array = np.array(all_st_clean, dtype=object)  # Use dtype=object for mixed-size arrays
    all_starttimes_array = np.array(all_starttimes)

    # Define the file path
    output_path = 'st_clean_2020-07-08T081938.npz'

    # Save both st_clean arrays and start times using np.savez
    np.savez(output_path, st_clean=all_st_clean_array, starttimes=all_starttimes_array)
    print(f"All st_clean arrays and starttimes saved to {output_path}")

    # End Timer
    end_time = time.time()

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")