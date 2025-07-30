#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:03:56 2024

@author: rachelwillis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer
import gpxpy
import rasterio
from rasterio.windows import from_bounds
from matplotlib.collections import LineCollection

# --- Plotting Style ---
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})

# --- File Paths ---
bottom_path = "../data/cable_layout/chmiel_gps_bottomcable.xlsx"
middle_path = "../data/cable_layout/chmiel_gps_middlecable.xlsx"
top_path = "../data/cable_layout/chmiel_gps_topcable_geophones.xlsx"
lower_csv = "../data/cable_layout/bowden_gps_lowercable.csv"
gpx_lower_path = "../data/cable_layout/gajek_gps_track_lowe_cable_only_out_of_glacier_part.gpx"
gpx_mid_path = "../data/cable_layout/gajek_gps_track_middle_drum_full_length.gpx"
gpx_em_path = "../data/cable_layout/Martins_Wegpunkte_04-JUL-20.gpx"
tiff_path = "../data/cable_layout/2020-01-05-00_00_2020-07-05-23_59_Sentinel-2_L1C_Custom_script.tiff"

# --- Functions ---
def transform_coords(easting, northing, crs_from='epsg:21781', crs_to='epsg:4326'):
    transformer = Transformer.from_crs(crs_from, crs_to)
    return zip(*[transformer.transform(e, n) for e, n in zip(easting, northing)])

def parse_gpx_points(gpx_path):
    with open(gpx_path, 'r') as f:
        gpx = gpxpy.parse(f)
    lat, lon = [], []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                lat.append(point.latitude)
                lon.append(point.longitude)
    return lat, lon

def parse_gpx_waypoints(gpx_path):
    with open(gpx_path, 'r') as f:
        gpx = gpxpy.parse(f)
    lat, lon, elev = [], [], []
    for wp in gpx.waypoints:
        lat.append(wp.latitude)
        lon.append(wp.longitude)
        elev.append(wp.elevation)
    return lat, lon, elev

# --- Load GPS Data ---
gps_bottom = pd.read_excel(bottom_path, engine='openpyxl')
gps_middle = pd.read_excel(middle_path, engine='openpyxl')
gps_top = pd.read_excel(top_path, engine='openpyxl')

bottom_lat, bottom_lon = transform_coords(gps_bottom['Unnamed: 3'], gps_bottom['Unnamed: 4'])
middle_elev = gps_middle['Coordinates'].dropna().to_list()
east_middle = gps_middle['Unnamed: 3'].dropna()
north_middle = gps_middle['Unnamed: 4'].dropna()
middle_lat, middle_lon = transform_coords(east_middle, north_middle)

middle_lat = list(middle_lat)
middle_lon = list(middle_lon)
middle_elev = list(middle_elev)

middle_lat.pop(9)
middle_lon.pop(9)
middle_elev.pop(9)

top_elev = gps_top['Coordinates']
east_top = gps_top['Unnamed: 3']
north_top = gps_top['Unnamed: 4']

# --- Combine all east/north for cable ---
east_tot = list(gps_bottom['Unnamed: 3']) + list(east_middle) + list(east_top)
north_tot = list(gps_bottom['Unnamed: 4']) + list(north_middle) + list(north_top)

# --- Parse GPX Files ---
lower_lat, lower_lon = parse_gpx_points(gpx_lower_path)
mid_lat, mid_lon = parse_gpx_points(gpx_mid_path)
em_lat, em_lon, em_elev = parse_gpx_waypoints(gpx_em_path)

# --- Lower Cable Elevation ---
lower_df = pd.read_csv(lower_csv, header=None)
interp_lower_elev = np.linspace(middle_elev[-1], list(gps_bottom['Coordinates'])[0], num=len(lower_df))

# --- Combine all data ---
tot_lon = em_lon + middle_lon + list(lower_df[2]) + list(bottom_lon)
tot_lat = em_lat + middle_lat + list(lower_df[1]) + list(bottom_lat)
tot_elev = em_elev + middle_elev + list(interp_lower_elev) + list(gps_bottom['Coordinates'])

# --- Plot Background Raster ---
lat_min, lat_max = 46.57, 46.64
lon_min, lon_max = 8.36, 8.42

with rasterio.open(tiff_path) as src:
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    left, bottom = transformer.transform(lon_min, lat_min)
    right, top = transformer.transform(lon_max, lat_max)
    window = from_bounds(left, bottom, right, top, transform=src.transform)
    tiff_crop = np.moveaxis(src.read([1, 2, 3], window=window), 0, -1)
    crop_transform = src.window_transform(window)

# --- Project Coordinates ---
transformer = Transformer.from_crs("epsg:4326", src.crs, always_xy=True)
cablex_proj, cabley_proj = transformer.transform(tot_lon, tot_lat)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(12, 12))
extent = (left, right, bottom, top)
ax.imshow(tiff_crop, extent=extent)

points = np.array([cablex_proj, cabley_proj]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap='plasma_r', norm=plt.Normalize(vmin=min(tot_elev), vmax=max(tot_elev)))
lc.set_array(np.array(tot_elev[:-1]))
lc.set_linewidth(5)
line = ax.add_collection(lc)

cb = fig.colorbar(line, ax=ax, orientation="horizontal", pad=0.08, aspect=40, shrink=0.62)
cb.set_label("Elevation [m]")
ax.set_xlabel("Longitude [°]")
ax.set_ylabel("Latitude [°]")

# Optional labels
subset_x = cablex_proj[4:-32]
subset_y = cabley_proj[4:-32]
label_indices = [0, len(subset_x)//3, 2*len(subset_x)//3, len(subset_x)-1]
labels = [1500, 1000, 500, 0]
for idx, label in zip(label_indices, labels):
    ax.text(subset_x[idx], subset_y[idx], str(label), color='black',
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.7))

plt.tight_layout()
#plt.savefig("cable_layout.pdf", format='pdf', bbox_inches='tight')
plt.show()
