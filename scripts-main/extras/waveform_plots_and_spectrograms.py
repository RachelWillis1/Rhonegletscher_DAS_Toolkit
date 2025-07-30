#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 07:36:18 2025

@author: rachelwillis
"""

import time
import numpy as np
from scipy.signal import sosfilt, hann, spectrogram
from obspy import UTCDateTime
from obspy import Trace, Stream, UTCDateTime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import dask
import dask.delayed

import par_file as pf
from readers import das_reader as reader
import preprocessing_functions as pp


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
        
        # p_arrivals = [11.42, 11.41, 11.40, 11.39, 11.38, 11.37, 11.38, 11.39, 11.40, 11.41]
        # p_arrivals = [(p - 11.25) * 220 for p in p_arrivals]
        # s_arrivals = [11.50, 11.48, 11.46, 11.44, 11.42, 11.40, 11.42, 11.44, 11.46, 11.48]
        # s_arrivals = [(s - 11.25) * 220 for s in s_arrivals]

        # Features Extraction
        for i in range(0, st_clean.shape[0], int(0.5 * (st_clean.shape[0] / 30))):

            ### === 1. PLOT WAVEFORMS === ###
            fig_wave, ax_wave = plt.subplots(figsize=(10, 13))  # Adjust figure size for waveforms
            
            for j in range(1250,1350,10): #range(0, st_clean.shape[1], 100):
                # Plot the waveforms
                ax_wave.plot(20 * st_clean[int(i):int(i+int(0.5 * (st_clean.shape[0]/30))), j] + (j/10), '-k')
            
            # Configure waveform plot
            channel_indices = np.arange(1250, 1350, 10) #np.arange(0, st_clean.shape[1], 100)
            ax_wave.set_yticks(channel_indices / 10)
            ax_wave.set_yticklabels([f"{c}" for c in channel_indices])
            
            # # Plot P and S arrival trends as lines
            # ax_wave.plot(p_arrivals, channel_indices / 10, '--b', label="P-wave Arrivals")  # Blue dashed line for P
            # ax_wave.plot(s_arrivals, channel_indices / 10, '--g', label="S-wave Arrivals")  # Green dashed line for S
            
            ax_wave.set_xticks([0, 55, 110])
            ax_wave.set_xticklabels([i/220, i/220+0.25, i/220+0.5])
            
            ax_wave.set_xlabel("Time (s)")
            ax_wave.set_ylabel("Channel")
            ax_wave.set_title(f"{str(starttime)[:-4]}")
            
            # Save the waveform figure
            # plt.savefig(f'/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/Rhone_Glacier_AWS_Final/AWS_Codes/codes/output/wvfm_plts_0.5sec/waveforms_{starttime}_ch_1350_1450_{i/220+0.25:.2f}_{i/220+0.75:.2f}_sec.png')
            plt.show()
            plt.close(fig_wave)

            # for j in range(0, st_clean.shape[1], 100):
                
            #     # data_fk, f_axis, k_axis = fkr_spectra(st_clean[i:i+int(2 * (st_clean.shape[0]/30)), j:j+100])
            #     data_chunk = st_clean[i:i+int(2 * (st_clean.shape[0]/30)),j]
            #     # plotFkSpectra(data_fk, f_axis, k_axis)
                

            #     # Define metadata
            #     trace_id = f"CH_{j}"  # Example channel name
            #     start_time = starttime + i / sampling_rate  # Adjust starttime based on index
                
            #     # Create an ObsPy Trace
            #     trace = Trace(data=data_chunk.astype(np.float32))  # Ensure correct dtype
            #     trace.stats.network = "XX"  # Custom network code
            #     trace.stats.station = "DAS"  # Station name
            #     trace.stats.channel = f"CH{j}"  # Assign channel name
            #     trace.stats.sampling_rate = sampling_rate  # Sampling rate
            #     trace.stats.starttime = UTCDateTime(start_time)  # Start time
                
            #     # Create an ObsPy Stream and add the trace
            #     stream = Stream(traces=[trace])
                
            #     fig = stream.spectrogram(show=False, log=True, title=str(stream[0].stats.starttime))
            #     fig = fig[0]
            #     ax = fig.axes[0]
            #     ax.set_xlim(0.026, 0.407)
            #     ax.set_ylim(10, 500)
            #     xticks = [0.026, 0.2165, 0.407]  # Modify based on your data
            #     ax.set_xticks(xticks)
            #     ax.set_xticklabels([i/220, i/220+1, i/220+2])  # Custom labels with seconds

            #     mappable = ax.collections[0]
            #     plt.colorbar(mappable=mappable, ax=ax)
            #     plt.show()
                
                # # Example Usage
                # fs = 1000  # Sampling frequency (Hz)

                # # Compute spectrogram
                # freqs, times, Sxx = compute_spectrogram_fft(data_chunk, fs=fs, nfft=144, overlap=144-12)

                # # Plot Spectrogram
                # plt.figure(figsize=(10, 6))
                # plt.pcolormesh(times, freqs, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
                # plt.ylabel("Frequency (Hz)")
                # plt.xlabel("Time (s)")
                # plt.ylim(10,90)
                # plt.colorbar(label="Power (dB)")
                # plt.yscale("log")
                # plt.title("Spectrogram using FFT")
                # plt.show()
                
                
                
                # fig_spec, ax_spec = plt.subplots(figsize=(6.4, 4.8))  # Adjust figure size for spectrogram
            
                # # Compute spectrogram for the current channel
                # f, t, Sxx = spectrogram(st_clean[i:i+int(2 * (st_clean.shape[0]/30)), j],
                #                         fs=220, nperseg=32, noverlap=16)
            
                # # Convert power spectrum to decibels
                # Sxx_db = 10 * np.log10(Sxx + 1e-10)
            
                # # Plot spectrogram with log scale
                # mesh = ax_spec.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis',
                #                           vmin=-60, vmax=-80)
            
                # # Set y-axis to logarithmic scale
                # ax_spec.set_yscale('log')
                # ax_spec.set_ylim([10, 110])  # Focus on active frequency range
            
                # # Format y-axis labels to display frequency properly
                # ax_spec.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
                # ax_spec.yaxis.set_minor_locator(ticker.NullLocator())  # Remove minor ticks
            
                # # Set x-axis tick labels to match time scaling
                # ax_spec.set_xticks([t[0], t[int(len(t)/2)], t[-1]])
                # ax_spec.set_xticklabels([i/220, i/220+1, i/220+2])
            
                # # Configure spectrogram plot
                # ax_spec.set_ylabel("Frequency (Hz)")
                # ax_spec.set_xlabel("Time (s)")
                # ax_spec.set_title(f"{str(starttime)[:-4]}")  # Only date-time in title
            
                # # Add annotation for channel number in upper right corner
                # ax_spec.text(0.97, 0.97, f"Ch {j}", transform=ax_spec.transAxes,
                #              ha='right', va='top', color='black',  # White text
                #              bbox=dict(facecolor='white', edgecolor='white', alpha=0.8, boxstyle='square,pad=0.3'))
            
                # # Add colorbar
                # cbar = plt.colorbar(mesh, ax=ax_spec)
                # cbar.set_label("Power Spectral Density (dB/Hz)")
            
                # # Save the spectrogram figure
                # plt.savefig(f'/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/Rhone_Glacier_AWS_Final/AWS_Codes/codes/output/spectrograms/spectrogram_{starttime}_ch{j}_{i/220:.2f}_{i/220+2:.2f}_sec.png')
                # plt.show()
                # plt.close(fig_spec)