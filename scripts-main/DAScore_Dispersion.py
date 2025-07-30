#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 08:01:35 2025

@author: rachelwillis
"""

import time
import numpy as np
import dascore as dc
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar


# # ============================
# #        INITIAL SETUP
# # ============================

# Start Timer
start_time = time.time()

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 16})

# ============================
#      DASCore Dispersion
# ============================

def write_to_dascore_spool(arr2d, data_name, fs):
    """
    writes a 2d array into the dascore spool format

    Parameters
    ----------
    arr2d: 2d array
        The 2d array of the shot gather slice
    data_name: String
        The name of the input 2d array
    fs: Int
        Original sampling frequency of raw data

    Returns
    -------
    pa: DASCore Patch
        The Patch format of the input array for use with DASCore
    """
    #attributes/metadata for patch
    attrs = dict(
        category = 'DAS',
        id = data_name
    )

    time_start = dc.to_datetime64("2002-02-01") #actual time start isn't necessary for our application of spool
    time_step = dc.to_timedelta64(1 / fs)
    time = time_start + np.arange(arr2d.shape[1]) * time_step

    distance_start = 0
    distance_step = 5
    distance = distance_start + np.arange(arr2d.shape[0]) * distance_step

    coords = dict(distance = distance, time=time)

    dims = ('distance', 'time')

    pa = dc.Patch(data = arr2d, coords = coords, attrs = attrs, dims = dims)
    return pa


# # ============================
# #        MAIN EXECUTION
# # ============================

test = []

loaded_data = np.load('st_clean_2020-07-07T153708_2020-07-07T154038.npz', allow_pickle=True)

st_clean = loaded_data['st_clean']
starttimes = loaded_data['starttimes']

print(f"Loaded st_clean shape: {st_clean.shape}")
print(f"Loaded start times: {starttimes}")

min_ch = 220
max_ch = 265

data = st_clean[5].T
data = data[min_ch:max_ch, 1380:1440] #channel, time
data = data.astype(np.float32)
starttime = starttimes[5]
patch = write_to_dascore_spool(data, f'{starttime}', 220)

ax1 = patch.viz.waterfall(show=False, cmap="seismic")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Distance (m)")
plt.savefig(f"waveform_waterfall_{starttime}_ch_{min_ch}_ {max_ch}.png", dpi=300, bbox_inches='tight')
# plt.show()

# Compute dispersion phase shift
disp_patch = patch.dispersion_phase_shift(np.arange(1000, 4000, 1), approx_resolution=0.1, approx_freq=[10, 90])

ax2 = disp_patch.viz.waterfall(show=False, cmap="viridis")
ax2.set_xlim(10, 90)
ax2.set_ylim(1000, 4000)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Velocity (m/s)")
ax2.text(105, 2200, "Amplitude", color="black", fontsize=16, ha='center', rotation=90)
plt.savefig(f"waveform_waterfall_disp_{starttime}_ch_{min_ch}_ {max_ch}.png", dpi=300, bbox_inches='tight')
plt.show()
