#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:48:27 2024

@author: rachelwillis
"""

import time
import numpy as np
import pandas as pd
from scipy.signal import iirfilter, zpk2sos, sosfilt, hann, sosfreqz
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import dask
import dask.delayed

import par_file as pf
import plotting_functions as plot
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
#          STA/LTA
# ============================
        
def stack_sta_lta_catalogue(st_clean, channel_index, time_index, startCh=0, endCh=500):
    """
    Linear stacking of reverse STA/LTA for DAS data.

    Parameters
    ----------
    st_clean : np.ndarray
        Preprocessed DAS data.
    channel_index : int
        Index of the current channel.
    time_index : int
        Index of the current time segment.

    Returns
    -------
    best_event : dict
        Dictionary containing only the highest SNR trigger for each (time_index, channel).
    """
    df_dec = pf.fs / pf.dec_factor
    npts = st_clean.shape[0]
    
    best_event = None  # Track the best event
    
    for i in range(startCh, endCh, pf.noverlap_sub):
        if (endCh - i) < pf.nr_sub_windows:
            continue  # Skip if not enough channels
        
        # Compute STA/LTA characteristic function
        cft = np.zeros(npts)
        for j in range(i, i + pf.nr_sub_windows):
            cft += recursive_sta_lta(st_clean[:, j], int(pf.sta_len * df_dec), int(pf.lta_len * df_dec))
        cft /= pf.nr_sub_windows

        # Find trigger events
        on_off = trigger_onset(cft, pf.trigger_thresh, pf.detrigger_thresh)
        if len(on_off) == 0:
            continue  # Skip if no triggers

        for event in range(on_off.shape[0]):
            trigger_on = int(on_off[event, 0])
            trigger_off = int(on_off[event, 1])
            if trigger_on == trigger_off:
                continue  # Skip invalid events

            event_SNR = float(cft[trigger_on:trigger_off].max())
            event_data = {'file_start': str(starttime), 'time': time_index/220, 'channel': channel_index, 'SNR': event_SNR}

            # Store the event only if it's the highest SNR for this (time_index, channel)
            if best_event is None or event_SNR > best_event['SNR']:
                best_event = event_data

    return best_event  # Return the best event found


# ============================
#        MAIN EXECUTION
# ============================

if __name__ == '__main__':
    catalogue_df = pd.DataFrame(columns=pf.cols)

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

        # Event Extraction
        for i, time_index in enumerate(range(0, st_clean.shape[0], 660)):
            noise_filt_agc = st_clean[time_index:time_index + 660, :]

            for startCh in range(0, pf.nrTr - 50 - 50, 500):
                best_event = stack_sta_lta_catalogue(noise_filt_agc[:, startCh:startCh + 500], channel_index=startCh, time_index=time_index)
                
                if best_event:
                    catalogue_df = pd.concat([catalogue_df, pd.DataFrame([best_event])], ignore_index=True)

                    # # Save Plot
                    # plot_filename = f"{pf.output_plt_dir}{starttime}_ch_{startCh}_{startCh+500}_t_{time_index/220}_{(time_index+660)/220}.png"
                    # plot.plotChRange(noise_filt_agc[:, startCh:startCh + 500], time_index, startCh=startCh, endCh=startCh + 500, 
                    #                          fs=250, clipPerc=99.97, title=str(starttime)[:-4], outfile=plot_filename)

    # Save the Final Catalogue
    catalogue_df.to_csv(pf.sta_lata_output_file, index=False)


# ============================
#        COMPLETION
# ============================

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Processing complete. Total time elapsed: {elapsed_time:.2f} seconds")

