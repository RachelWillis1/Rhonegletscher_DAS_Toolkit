#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:04:54 2024

@author: rachelwillis
"""

import json
import numpy as np
import glob
from obspy import UTCDateTime
from scipy.signal import iirfilter, zpk2sos, sosfilt, sosfreqz
from scipy.signal.windows import hann

# ============================
#        GENERAL SETTINGS
# ============================

# Load channels list
#with open("channels_to_read_new.json", 'r') as f:
#    channels_to_read_new = json.load(f)

# File Paths
#data_dir = ("/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/Rhone_Glacier_AWS_Final/20200707_140000.000-20200707_160000.000/codes/input/DAS_data/") #("../input/DAS_data/")
#ml_model = "/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/Rhone_Glacier_AWS_Final/AWS_Codes/codes/random_forest_model.pkl"
# ============================
# Define Target Time Range
# ============================
#target_starttime = UTCDateTime("2020-07-07T15:37:00")  # Define your desired start time
#target_endtime = UTCDateTime("2020-07-07T15:41:00")    # Define your desired end time

#sta_lata_output_file = "../output/sta_lta_event_catalogue.csv"
#output_plt_dir = "../output/event_pngs/"

# Load channels list
with open("/home/ec2-user/Rhonegletscher_DAS_ML_Toolkit/scripts-main/channels_to_read_new.json", 'r') as f:
    channels_to_read_new = json.load(f)

# File Paths
data_dir = ("/home/ec2-user/Rhonegletscher_DAS_ML_Toolkit/input/DAS_data/")
ml_model = "/home/ec2-user/Rhonegletscher_DAS_ML_Toolkit/random_forest_model.pkl"

output_dir = ("/home/ec2-user/Rhonegletscher_DAS_ML_Toolkit/output/")
output_plt_dir = ("/home/ec2-user/Rhonegletscher_DAS_ML_Toolkit/output/event_pngs/")
sta_lata_output_file = "/home/ec2-user/Rhonegletscher_DAS_ML_Toolkit/output/sta_lta_event_catalogue.csv"

# Load and Sort File List
nr_files_to_load = 3
nr_files_to_process = nr_files_to_load - 2
files_list = sorted(glob.glob(data_dir + "*.hdf5"))
nr_files = len(files_list)


# ============================
#        SAMPLING SETTINGS
# ============================

# Sampling Parameters
file_len = 30 # Length of file in seconds
fs_orig = 1000  # Original Sampling Frequency (Hz)
npts = int(nr_files_to_load * fs_orig * file_len)  # Number of total points
dec_factor = 5  # Downsampling Factor
fs = fs_orig // dec_factor  # New Sampling Frequency after Downsampling

# Time Windowing
len_ave_window = int(5*fs)  # Long window length
noverlap_ave = int(2*fs)  # Long window overlap
len_sub_window = int(0.6*fs)  # Short window length
noverlap_sub = len_sub_window // 2  # Short window overlap
nr_sub_windows = len_ave_window // noverlap_sub - 1  # Number of short windows

# n_twin = (((nr_files-nr_files_to_load) // nr_files_to_process)
#           * int(nr_files_to_process * 30 * fs / noverlap_ave)) # Number of time windows

# Define Start and End Sample to avoid tapered data
start_sample = int(file_len * fs)
end_sample = int((nr_files_to_load - 1) * file_len * fs + (len_ave_window - noverlap_ave))

# Channel Settings
startCh = 400
endCh = 2000
channels_example = list(range(0, 500, 4))

nstations = len(channels_example)
nfreqs = len_sub_window // 2 + 1


# ============================
#            TAPER
# ============================

# Taper Settings (To Reduce Edge Effects)
max_perc = 0.05  # 5% taper at the start and end of traces

# Time-Domain Taper
wlen = int(max_perc * int(nr_files_to_load * fs * file_len))
taper_sides = hann(2 * wlen + 1)
taper = np.hstack((taper_sides[:wlen], np.ones(npts-2*wlen), taper_sides[len(taper_sides)-wlen:]))

# ============================
#        BANDPASS FILTER
# ============================

# Bandpass Filter Parameters
corners = 4
freqmin = 10  # Minimum frequency
freqmax = 90  # Maximum frequency

# Bandpass Filter for Time-Domain Processing
fe = 0.5 * fs_orig
low = freqmin / fe
high = freqmax / fe
z, p, k = iirfilter(corners, [low, high], btype='band', ftype='butter', output='zpk')
sos = zpk2sos(z, p, k)

# Bandpass Filter for Downsampled Data
fe2 = 0.5 * fs
low2 = freqmin / fe2
high2 = freqmax / fe2
z2, p2, k2 = iirfilter(corners, [low2, high2], btype='band', ftype='butter', output='zpk')
sos2 = zpk2sos(z2, p2, k2)

# Frequency Domain Taper
f_axis_l = np.fft.rfftfreq(int(nr_files_to_load * file_len * fs), d=1 / fs)
w, h = sosfreqz(sos2, len(f_axis_l))
f_taper = np.abs(h)


# ============================
#        SPATIAL FILTER
# ============================

# Spatial Highpass Filter
nrTr = endCh - startCh  # Number of Traces (Total Channels Used)
df_x = 1/4  # Channel Spacing (Inverse of Sampling Rate)
corners_x = 8  # Filter Order
flow = 0.002  # High-pass Cutoff Frequency

# Taper for Spatial Filtering
wlen_x = int(max_perc * nrTr)
taper_sides_x = hann(2 * wlen_x + 1)
taper_x = np.hstack((taper_sides_x[:wlen_x], np.ones(nrTr - 2 * wlen_x), taper_sides_x[len(taper_sides_x) - wlen_x:]))

# Highpass Filter for Spatial Filtering
fe_x = 0.5 * df_x
f = flow / fe_x
zx, px, kx = iirfilter(corners_x, f, btype='highpass', ftype='butter', output='zpk')
sos_x = zpk2sos(zx, px, kx)


# ============================
#            STA/LTA
# ============================

# STA/LTA Triggering Parameters
sta_len = 0.083  # Short-Term Average Window Length (seconds)
lta_len = 0.5  # Long-Term Average Window Length (seconds)
trigger_thresh = 3.4 #3.4 # 4.0  # Trigger Threshold
detrigger_thresh = 3.1 #3.1 #3.8  # Detrigger Threshold

# STA/LTA Catalogue Settings
cols = ['file_start', 'time', 'channel', 'SNR']
