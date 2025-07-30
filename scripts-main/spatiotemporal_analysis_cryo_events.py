#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:10:40 2024

@author: rachelwillis
"""

import os
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta


# Set Times New Roman as the default font
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams.update({'font.size': 20})  # Increase font size for readability

# ==== Define Paths ====
meta_data_dir = "../../AWS_Output_Final/meta_data_coherency"
rf_pred_dir = "../../AWS_Output_Final/ref_pred"

# Get sorted file lists
meta_data_files = sorted([f for f in os.listdir(meta_data_dir) if f.endswith(".npy")])
rf_pred_files = sorted([f for f in os.listdir(rf_pred_dir) if f.endswith(".npy")])

# Containers for all data
all_space = []
all_time = []
all_rf_preds = []

# Function to extract timestamp from filenames
def extract_date_time(file_path):
    match = re.search(r"UTC_(\d{8})_(\d{6})\.\d{3}", file_path)
    if match:
        date_part = match.group(1)  # YYYYMMDD
        time_part = match.group(2)  # HHMMSS
        formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"  # YYYY-MM-DD
        formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"  # HH:MM:SS
        return f"{formatted_date} {formatted_time}"
    return None

# Load metadata and RF predictions
for meta_file, rf_file in zip(meta_data_files, rf_pred_files):
    meta_data_path = os.path.join(meta_data_dir, meta_file)
    rf_pred_path = os.path.join(rf_pred_dir, rf_file)

    # Load metadata and predictions
    meta_data = np.load(meta_data_path, allow_pickle=True)
    rf_pred = np.load(rf_pred_path, allow_pickle=True)

    # Extract relevant columns
    space = meta_data[:, 0].astype(float)
    time = meta_data[:, 1].astype(float)
    starttime_filenames = meta_data[:, 2]

    # Extract and convert timestamps
    extracted_start_times = np.array([extract_date_time(val) for val in starttime_filenames if extract_date_time(val) is not None])
    start_times = np.array([datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in extracted_start_times])

    # Adjust times using metadata time offsets
    adjusted_times = start_times + np.array([timedelta(seconds=t) for t in time])

    # Append to lists
    all_space.extend(space)
    all_time.extend(adjusted_times)
    all_rf_preds.extend(rf_pred)

# Convert lists to NumPy arrays
all_space = np.array(all_space)
all_time = np.array(all_time)
all_rf_preds = np.array(all_rf_preds)

# ==== Create Heatmap ====
print("\n Generating Cluster 0 Frequency Heatmap...")

# Define **time bins** (every 2 hours)
start_time = min(all_time)
end_time = max(all_time)
time_bins = [start_time + timedelta(hours=2 * i) for i in range((end_time - start_time).days * 12 + 1)]

# Define **space bins** (channel groups)
space_bins = np.arange(0, 1500 + 1, 500)  # Adjust based on your data

# Initialize 2D heatmap grid
heatmap = np.zeros((len(space_bins) - 1, len(time_bins) - 1))

# Fill in heatmap
for i in range(len(time_bins) - 1):
    for j in range(len(space_bins) - 1):
        # Mask for data within the bin
        mask = (
            (all_time >= time_bins[i]) & (all_time < time_bins[i + 1]) &
            (all_space >= space_bins[j]) & (all_space < space_bins[j + 1]) &
            (all_rf_preds == 0)  # Only count Cluster 0 or 1
        )
        heatmap[j, i] = np.sum(mask)  # Count occurrences

# ==== Plot Heatmap ====
fig, ax = plt.subplots(figsize=(22, 9))

# Define colormap
cmap = plt.cm.Reds  # Using Reds for intensity visualization
norm = mcolors.Normalize(vmin=0, vmax=np.max(heatmap))  # Normalize color scale

# Create heatmap plot
pcm = ax.pcolormesh(time_bins, space_bins, heatmap, cmap=cmap, norm=norm, shading="auto")

# Add colorbar
cbar = plt.colorbar(pcm, pad=0.02)
cbar.set_label("Cluster 0 Frequency")

# Format x-axis (time labels)
num_ticks = 6  # Number of evenly spaced time labels
x_tick_indices = np.linspace(0, len(time_bins) - 2, num_ticks, dtype=int)
x_tick_labels = [time_bins[i].strftime("%Y-%m-%d") for i in x_tick_indices]

ax.set_xticks([time_bins[i] for i in x_tick_indices])
ax.set_xticklabels(x_tick_labels, ha="center")

ax.xaxis.set_tick_params(pad=20)

# Set y-axis labels (channel bins)
ax.set_yticks(space_bins)
ax.set_yticklabels([f"{int(s)}" for s in space_bins])

# Labels and title
ax.set_xlabel("Time")
ax.set_ylabel("Channel")

# Rotate x-axis labels
# plt.xticks(rotation=25, ha="right")

# Save or Show
plt.tight_layout()
plt.show()
    
    
    