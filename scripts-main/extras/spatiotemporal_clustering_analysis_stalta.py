#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:39:52 2024

@author: rachelwillis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates


# Set Times New Roman as the default font
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams.update({'font.size': 20})


coh_agg = np.load('/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/Rhone_Glacier_AWS_Final/Rhone_Glacier_STA_LTA_Long_Run2/codes/output/coh_agg_stalta_7clusters.npy')
catalogue_df = pd.read_csv('/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/Rhone_Glacier_AWS_Final/Rhone_Glacier_STA_LTA_Long_Run2/codes/output/catalogue.csv')

catalogue_df['starttime'] = pd.to_datetime(catalogue_df['starttime'])
event_times = catalogue_df['starttime'] + pd.to_timedelta((catalogue_df['time_vals'] / 220), unit='s')

n_clusters = 7

# Extract date for the title
date_for_title = event_times.iloc[0].strftime('%Y-%m-%d %H:%M:%S')

# Convert event_times to matplotlib date format
event_times = mdates.date2num(event_times)

space_vals = catalogue_df['space_vals']

# Use Viridis colormap from matplotlib
cmap = plt.cm.viridis

# Set discrete boundaries for the colorbar
boundaries = np.linspace(0, n_clusters, n_clusters + 1)
norm = mcolors.BoundaryNorm(boundaries, cmap.N)

# Plotting the clusters over time and distance
plt.figure(figsize=(10, 6))

# Plot vertical lines for each event with appropriate color for each cluster
for i in range(len(event_times)):
    plt.vlines(event_times[i], catalogue_df['space_vals'][i], catalogue_df['space_vals'][i] + 500,
               color=cmap(norm(coh_agg[i])), linewidth=10)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(coh_agg)  # Set the array for ScalarMappable

# Create the colorbar with discrete ticks at boundaries
cbar = plt.colorbar(sm, ticks=np.arange(0, n_clusters + 1))  # Discrete ticks

# Adjust tick labels to be in the middle of the segments
midpoints = (boundaries[:-1] + boundaries[1:]) / 2  # Calculate midpoints of each segment
cbar.set_ticks(midpoints)  # Set the ticks to the midpoints
cbar.ax.set_yticklabels([f'{i}' for i in np.arange(0, n_clusters + 1)])
cbar.set_label('Clusters')

# Format x-axis as dates and improve readability
plt.gca().xaxis_date()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gcf().autofmt_xdate(rotation=45)  # Auto-format and rotate labels

plt.title(f'Start Date: {date_for_title}')
plt.xlabel('Timestamp')
plt.xlim(event_times.min(), event_times.max())
plt.ylabel('Channel number')
plt.ylim(0,1500)
plt.show()

# Count the occurrences of each number in coh_agg (0, 1, 2, 3, 4, 5)
counts = np.bincount(coh_agg)

# Print the results
for i in range(len(counts)):
    print(f'Number {i} occurs {counts[i]} times.')