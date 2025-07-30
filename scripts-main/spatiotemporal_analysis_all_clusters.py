#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:10:40 2024

@author: rachelwillis
"""

import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Set Times New Roman and font size ---
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams.update({'font.size': 25})

# --- Load Feature Data ---
feature_file = "/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/Rhone_Glacier_AWS_Final/AWS_Codes_For_Cloud/data/meta_data_coherency/20200708_070000.000-20200708_090000.000/meta_data_coherency.npy"
features = np.load(feature_file, allow_pickle=True)

# --- Load Trained Model & Predict Clusters ---
rf_model_loaded = joblib.load("/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/Rhone_Glacier_AWS_Final/AWS_Codes_For_Cloud/random_forest_model.pkl")
rf_pred = rf_model_loaded.predict(features[:, 3:])

# --- Extract Space and Time ---
space = features[:, 0].astype(float)
time = features[:, 1].astype(float)

# --- Helper Function: Extract Date/Time from Filename ---
def extract_date_time(file_path):
    match = re.search(r"UTC_(\d{8})_(\d{6})\.\d{3}", file_path)
    if match:
        date_part = match.group(1)
        time_part = match.group(2)
        formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"
        formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
        return formatted_date, formatted_time
    return None, None

# --- Apply Formatting to Start Times ---
starttime_filenames = features[:, 2]
formatted_dates, formatted_times = zip(*[extract_date_time(val) for val in starttime_filenames])
unique_starttimes = np.unique(formatted_times)
unique_dates = np.unique(formatted_dates)
display_date = unique_dates[0] if len(unique_dates) == 1 else "Multiple Dates"

# --- Create Time Offsets for Each Start Time ---
time_offsets = {start: i * 35 for i, start in enumerate(unique_starttimes)}
adjusted_time = np.array([time[i] + time_offsets[start] for i, start in enumerate(formatted_times)])

# --- Define Time and Space Bins ---
time_bins = np.arange(min(adjusted_time), max(adjusted_time) + 3, 3)
space_bins = np.array([0, 500, 1000, 1500])

# --- Create Cluster Grid ---
cluster_grid = np.zeros((len(space_bins) - 1, len(time_bins) - 1))
for i in range(len(time_bins) - 1):
    for j in range(len(space_bins) - 1):
        mask = (
            (adjusted_time >= time_bins[i]) & (adjusted_time < time_bins[i + 1]) &
            (space >= space_bins[j]) & (space < space_bins[j + 1])
        )
        if np.any(mask):
            cluster_grid[j, i] = np.bincount(rf_pred[mask]).argmax()

# --- Custom Colormap for Clusters ---
cluster_colors = ['red', 'blue', 'white', 'yellow']  # 0=red, 1=blue, 2=white, 3=yellow
cmap = mcolors.ListedColormap(cluster_colors)
norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, 4, 1), ncolors=4)

# --- Plot ---
fig, ax = plt.subplots(figsize=(20, 7))
pcm = ax.pcolormesh(time_bins, space_bins, cluster_grid, cmap=cmap, norm=norm, shading="flat")

# --- Colorbar ---
cbar = plt.colorbar(pcm, ticks=[0, 1, 2, 3], pad=0.02)
cbar.set_label("Cluster Value")

# --- X-Ticks (Show 4 evenly spaced times) ---
num_ticks = 4
x_tick_indices = np.linspace(0, len(unique_starttimes) - 1, num_ticks, dtype=int)
x_tick_positions = [time_offsets[unique_starttimes[i]] for i in x_tick_indices]
x_tick_labels = [unique_starttimes[i] for i in x_tick_indices]
ax.set_xticks(x_tick_positions)
ax.set_xticklabels(x_tick_labels, ha="center")
ax.xaxis.set_tick_params(pad=15)

# --- Y-Ticks ---
ax.set_yticks([0, 500, 1000, 1500])
ax.set_yticklabels(["0", "500", "1000", "1500"])

# --- Labels & Date Annotation ---
ax.set_xlabel("Starttime")
ax.set_ylabel("Channel")
ax.text(0.99, 0.98, f"Date: {display_date}", transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

plt.show()


    
    
    
    