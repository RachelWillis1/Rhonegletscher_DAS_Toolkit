#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:27:10 2025

@author: rachelwillis
"""

import pandas as pd
import numpy as np
import glob
import re
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report

# ============================= #
# === Load STA/LTA Results ==== #
# ============================= #

clusters_path = "../data/MyClustersInfo_Expanded_CV_corrected.xlsx"
rf_predictions_path = "../data/rf_predictions.csv"

# Load clusters_info
clusters_df = pd.read_excel(clusters_path, engine="openpyxl")
clusters_df.rename(columns={'filename': 'file_start', 'space': 'channel'}, inplace=True)

# Function to extract and format datetime from filename
def extract_datetime_from_filename(filename):
    match = re.search(r'(\d{8}_\d{6}\.\d+)', filename)  # Extract YYYYMMDD_HHMMSS.sss
    if match:
        raw_datetime = match.group(1)
        dt_obj = datetime.strptime(raw_datetime, "%Y%m%d_%H%M%S.%f")
        formatted_dt = dt_obj.strftime("%Y-%m-%dT%H:%M:%S.%f")[:] + "Z"  # Trim to match precision
        return formatted_dt
    return None

# Apply extraction function
clusters_df['file_start'] = clusters_df['file_start'].apply(extract_datetime_from_filename)
clusters_df['time'] = clusters_df['time'].astype(float).round(3)
clusters_df['channel'] = clusters_df['channel'].astype(int)
clusters_df['file_start'] = clusters_df['file_start'].astype(str).str.strip()

# Function to process STA/LTA data (Upper or Lower)
def process_sta_lta(csv_pattern, label):
    print(f"\nProcessing {label} Threshold STA/LTA Dataset...")

    # Load CSV files
    csv_files = glob.glob(f"../data/sta_lta_csv_event_pngs/{csv_pattern}")  
    events_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Standardize types
    events_df['time'] = events_df['time'].astype(float).round(3)
    events_df['channel'] = events_df['channel'].astype(int)
    events_df['file_start'] = events_df['file_start'].astype(str).str.strip()

    # Merge with clusters_df
    events_df_filtered = events_df.merge(
        clusters_df[['file_start', 'time', 'channel', 'cluster']], 
        on=['file_start', 'time', 'channel'], 
        how='inner'
    )

    # Classification
    events_df_filtered['classification'] = events_df_filtered['cluster'].apply(
        lambda x: 'True Positive' if x in [0, 1] else 'False Positive'
    )

    # Find missing detections
    merged_df = clusters_df.merge(events_df_filtered[['file_start', 'time', 'channel']], 
                                  on=['file_start', 'time', 'channel'], 
                                  how='left', indicator=True)

    merged_df['classification'] = 'Unclassified'
    merged_df['classification'] = merged_df.apply(
        lambda row: 'True Negative' if row['cluster'] in [2, 3] and row['_merge'] == 'left_only'
        else ('False Negative' if row['cluster'] in [0, 1] and row['_merge'] == 'left_only' else row['classification']),
        axis=1
    )

    merged_df = merged_df.drop(columns=['_merge'])
    final_classified_df = pd.concat([events_df_filtered, merged_df], ignore_index=True)

    # Remove Unclassified Entries
    final_classified_df = final_classified_df[final_classified_df['classification'] != 'Unclassified']

    # Compute Metrics
    tp = (final_classified_df['classification'] == 'True Positive').sum()
    fp = (final_classified_df['classification'] == 'False Positive').sum()
    tn = (final_classified_df['classification'] == 'True Negative').sum()
    fn = (final_classified_df['classification'] == 'False Negative').sum()

    print(f"{label} - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    # Compute y_true and y_pred
    y_true = []
    y_pred = []

    for _, row in final_classified_df.iterrows():
        if row['classification'] == 'True Positive':
            y_true.append(1)
            y_pred.append(1)
        elif row['classification'] == 'False Positive':
            y_true.append(0)
            y_pred.append(1)
        elif row['classification'] == 'True Negative':
            y_true.append(0)
            y_pred.append(0)
        elif row['classification'] == 'False Negative':
            y_true.append(1)
            y_pred.append(0)

    # Convert to NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute Confusion Matrix and Classification Report
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    class_report = classification_report(y_true, y_pred, target_names=["Non Events", "Events"])

    return conf_matrix, class_report, (tp, fp, tn, fn)

# Process both STA/LTA datasets
upper_conf_matrix, upper_class_report, upper_metrics = process_sta_lta("*_upper_threshold.csv", "Upper")
lower_conf_matrix, lower_class_report, lower_metrics = process_sta_lta("*_lower_threshold.csv", "Lower")

# ============================= #
# === Load Random Forest Predictions ==== #
# ============================= #

rf_df = pd.read_csv(rf_predictions_path)

# Apply function to `rf_df['filename']`
rf_df['file_start'] = rf_df['filename'].apply(extract_datetime_from_filename)
rf_df.rename(columns={'space': 'channel'}, inplace=True)

rf_df['time'] = rf_df['time'].astype(float).round(3)
rf_df['channel'] = rf_df['channel'].astype(int)

rf_df = rf_df.merge(clusters_df[['file_start', 'time', 'channel']], 
    on=['file_start', 'time', 'channel'], how='inner')

# Classify Random Forest Predictions
rf_df['classification'] = rf_df.apply(
    lambda row: 'True Positive' if row['actual_cluster'] in [0, 1] and row['predicted_cluster'] in [0, 1] else
    ('False Positive' if row['actual_cluster'] in [2, 3] and row['predicted_cluster'] in [0, 1] else
    ('True Negative' if row['actual_cluster'] in [2, 3] and row['predicted_cluster'] in [2, 3] else
    'False Negative')), axis=1)

# Compute Confusion Matrix & Classification Report for Random Forest
y_true_rf = rf_df['actual_cluster'].map(lambda x: 1 if x in [0, 1] else 0).to_numpy()
y_pred_rf = rf_df['predicted_cluster'].map(lambda x: 1 if x in [0, 1] else 0).to_numpy()

conf_matrix_rf = confusion_matrix(y_true_rf, y_pred_rf)
class_report_rf = classification_report(y_true_rf, y_pred_rf, target_names=["Non Events", "Events"])

# ============================= #
# === Print Comparison ==== #
# ============================= #

print("\nUpper STA/LTA Confusion Matrix:\n", upper_conf_matrix)
print("\nUpper STA/LTA Classification Report:\n", upper_class_report)

print("\nLower STA/LTA Confusion Matrix:\n", lower_conf_matrix)
print("\nLower STA/LTA Classification Report:\n", lower_class_report)

print("\nRandom Forest Confusion Matrix:\n", conf_matrix_rf)
print("\nRandom Forest Classification Report:\n", class_report_rf)

print("\nComparison between Upper STA/LTA, Lower STA/LTA, and Random Forest complete.")