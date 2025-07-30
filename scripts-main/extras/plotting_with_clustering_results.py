#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:47:26 2024

@author: rachelwillis
"""

import json
import numpy as np
import pandas as pd
import os
import glob
import time

from scipy.signal import iirfilter, zpk2sos, sosfilt, hann, sosfreqz, decimate
from obspy import UTCDateTime, Stream, Trace
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import dask
import dask.array as da
import dask.delayed
from dask.distributed import Client

from pydvs import coherency_functions
from pydvs import fk_functions
from readers import das_reader as reader
from par_file import *

#mpl.use('agg')

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

def preproc(tr_id, st_demean, dec_factor=5):
    """Taper, filter, and decimate along time-axis."""
    npts = len(st_demean[:, tr_id])
    wlen = int(max_perc * npts)
    taper_sides = hann(2*wlen+1)
    taper = np.hstack((taper_sides[:wlen], np.ones(npts-2*wlen), taper_sides[len(taper_sides)-wlen:]))
    tapered = st_demean[:, tr_id] * taper
    firstpass = sosfilt(sos, tapered)
    filtered = sosfilt(sos, firstpass[::-1])[::-1]
    downsampled = filtered[::dec_factor]
    filtered_white = coherency_functions.single_whiten_taper(downsampled, f_taper=f_taper, f_smooth=10)
    return filtered_white

def preproc_x(tr_t_id, st_whitened_agc):
    """Taper, filter, and decimate along space-axis."""
    tapered = st_whitened_agc[tr_t_id, :] * taper_x
    firstpass = sosfilt(sos_x, tapered)
    filtered = sosfilt(sos_x, firstpass[::-1])[::-1]
    return filtered



if __name__ == '__main__':
    
    coh_agg_results = np.load('/home/ec2-user/Rhone_Glacier/codes/output/20200707_074000.000-20200707_102000.000/coh_agg.npy')
#    coh_kmeans_results = np.load('/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/RG_Glacier_Updated/RG_Glacier/codes/output/coh_kmeans.npy')
    
#    vel_agg_results = np.load('/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/RG_Glacier_Updated/RG_Glacier/codes/output/velo_agg.npy')
#    vel_kmeans_results = np.load('/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/RG_Glacier_Updated/RG_Glacier/codes/output/velo_kmeans.npy')

    time_counter = 0
    for file_id in range(0, nr_files - nr_files_to_load, nr_files_to_process):
        
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

        # Initiate figure for each file
        fig, axs = plt.subplots(nrows=10, ncols=3, figsize=(12,24))
        plt.subplots_adjust(hspace=0.2, wspace=0.1)  # Adjust space between subplots

        # Remove time-mean from data
        time_mean = st[0].mean(axis=1).reshape(-1, 1)
        st_demean = st[0] - time_mean

        print("Starting parallelized pre-processing (filtering+whitening)...")

        # Use Dask's delayed functions for parallel processing
        delayed_results = [dask.delayed(preproc)(tr_id, st_demean) for tr_id in range(nrTr)]
        results = dask.compute(*delayed_results)  # Trigger the computation

        results_array = np.stack(results).T
        st_whitened = results_array[start_sample:end_sample, :]
        print('Complete! Quick check... Format of results: ({},{})'.format(
            st_whitened.shape[0], st_whitened.shape[1]))
        print('')
        st_whitened = st_whitened - st_whitened.mean(axis=1).reshape(-1,1) 
        st_whitened_agc, tosum = fk_functions.AGC(st_whitened, 400) 

        print("Starting parallelized pre-processing (high-pass space)...")

        # Use Dask's delayed functions for space-axis processing
        delayed_results1 = [dask.delayed(preproc_x)(tr_t_id, st_whitened_agc) for tr_t_id in range(st_whitened_agc.shape[0])]
        results1 = dask.compute(*delayed_results1)  # Trigger the computation

        results_array1 = np.stack(results1)
        st_clean = results_array1[:, 50:-50]
        print('Complete! Quick check... Format of results: ({},{})'.format(
            st_clean.shape[0], st_clean.shape[1]))
        
        # Determine frame color based on clustering results (example logic)
        cluster_labels = coh_agg_results[time_counter * 3:time_counter * 3 + 30]
        cluster_labels_reshaped = np.reshape(cluster_labels, (10, 3))
    
        print('Starting feature extraction...')
        for i,time_index in enumerate(range(0, st_whitened.shape[0]-2*noverlap_ave,
                                len_ave_window-noverlap_ave)):
            
            noise_filt_agc = st_clean[time_index:time_index+len_ave_window-noverlap_ave, :]
            
            # Plotting Spatio-temporal chunks. Dask cannot handle plotting!
            for j,start_ch_plt in enumerate(range(0, nrTr - 50 - 50, 500)):
                cluster_label = cluster_labels_reshaped[i, j]
                channels = list(range(start_ch_plt, start_ch_plt+500, 4))
                st_ave_window_plt = noise_filt_agc[:, channels]
                fk_functions.plotChRange(st_ave_window_plt, time_index, startCh=start_ch_plt, endCh=start_ch_plt+500, fs=fs,
                                          clipPerc=99.97, ax=axs[abs(i-9), j], frame_color=cluster_colors[cluster_label])  # Pass subplot axes to plot function)

            # Advance `time_counter` globally
            time_counter += 1
            
            
        # Add overall axis labels
        fig.text(0.5, 0.11, 'Channel number', ha='center', va='center')
        fig.text(0.07, 0.5, 'Time [s]', ha='center', va='center', rotation='vertical')
    
        # Add overall title
        fig.suptitle('Starttime: {}'.format(files_list[file_id][28:-15]), fontsize=16, y=0.90)
        
        # Remove inner axis numbers
        for k, ax in enumerate(axs.flat):
            row, col = divmod(k, axs.shape[1])
            # Remove x-ticks and y-ticks for inner subplots
            if row != axs.shape[0] - 1:  # Not the bottom row
                ax.set_xticks([])
                ax.set_xticklabels([])
            if col != 0:  # Not the leftmost column
                ax.set_yticks([])
                ax.set_yticklabels([])
                
        # Create custom legend handles
        legend_handles = [mpatches.Patch(color=color, label='Cluster {}'.format(label)) for label, color in cluster_colors.items()]

        # Add a legend to the overall figure
        plt.legend(handles=legend_handles, bbox_to_anchor=(1.6, 11.85), title='Cluster Assignment')

        # outfile = f"{files_list[file_id][28:-15]}.pdf"
        # plt.savefig(f"{files_list[file_id][60:-15]}.pdf", format='pdf', bbox_inches='tight')
        plt.show()
