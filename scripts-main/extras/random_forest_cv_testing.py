#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:32:39 2025

@author: rachelwillis
"""

from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score, classification_report, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV

# Set Times New Roman as the default font
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams.update({'font.size': 23})

# Paths to uploaded files
feature_files = [
    "../../data/hyperparameter_gridsearch/20200706_071820.361-20200706_091820.361/meta_data_coherency.npy",
    "../../data/hyperparameter_gridsearch/20200707_070000.000-20200707_090000.000/meta_data_coherency.npy",
    "../../data/hyperparameter_gridsearch/20200707_140000.000-20200707_160000.000/meta_data_coherency.npy",
    "../../data/hyperparameter_gridsearch/20200708_070000.000-20200708_090000.000/meta_data_coherency.npy",
    "../../data/hyperparameter_gridsearch/20200708_140000.000-20200708_160000.000/meta_data_coherency.npy",
    "../../data/hyperparameter_gridsearch/20200709_070000.000-20200709_090000.000/meta_data_coherency.npy",
    "../../data/hyperparameter_gridsearch/20200715_070000.000-20200715_090000.000/meta_data_coherency.npy",
    "../../data/hyperparameter_gridsearch/20200715_140000.000-20200715_160000.000/meta_data_coherency.npy",
    "../../data/hyperparameter_gridsearch/20200803_070000.000-20200803_090000.000/meta_data_coherency.npy"
]

clusters_path = "../../data/hyperparameter_gridsearch/MyClustersInfo_Expanded_CV_corrected.xlsx"

# Load clusters_info
clusters_info = pd.read_excel(clusters_path, engine="openpyxl")
if 'cluster' not in clusters_info.columns:
    raise ValueError("The data must contain a 'cluster' column.")

# Prepare clusters_info
clusters_info['filename'] = clusters_info['filename'].astype(str).str[-44:]  # Keep last 44 characters
clusters_info['time'] = clusters_info['time'].astype(float)
clusters_info['space'] = clusters_info['space'].astype(int)

# Combine all feature files
combined_features = []
for feature_file in feature_files:
    features = np.load(feature_file)

    # Convert the feature array to a DataFrame
    feature_columns = ['space', 'time', 'filename'] + [f'feature_{i}' for i in range(features.shape[1] - 3)]
    feature_df = pd.DataFrame(features, columns=feature_columns)

    # Prepare feature_df
    feature_df['filename'] = feature_df['filename'].astype(str).str[-44:]  # Keep last 44 characters
    feature_df['time'] = feature_df['time'].astype(float)
    feature_df['space'] = feature_df['space'].astype(int)

    combined_features.append(feature_df)

# Concatenate all feature DataFrames
all_features_df = pd.concat(combined_features, ignore_index=True)

# Filter clusters_info to exclude missing features
filtered_clusters_info = clusters_info.merge(
    all_features_df[['filename', 'time', 'space']],
    on=['filename', 'time', 'space'],
    how='inner'  # Keep only matches
)

ground_truth_labels = filtered_clusters_info['cluster']

# Filter features that match with filtered_clusters_info
matching_features = pd.merge(
    all_features_df,
    filtered_clusters_info[['filename', 'time', 'space']],
    on=['filename', 'time', 'space'],
    how='inner'  # Keep only matches
)

# Identify features in clusters_info that are not in feature_df
missing_features = clusters_info.merge(
    all_features_df[['filename', 'time', 'space']],
    on=['filename', 'time', 'space'],
    how='left',  # Left join to retain all `clusters_info` entries
    indicator=True  # Add a column to indicate presence in both dataframes
)

# Filter rows that are present in `clusters_info` but missing in `feature_df`
missing_features = missing_features[missing_features['_merge'] == 'left_only']

# Extract the filtered feature values
filtered_features = matching_features.drop(columns=['filename', 'time', 'space']).to_numpy()

# Initialize results storage
results = {}


# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(filtered_features, ground_truth_labels, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_split=2, 
    max_features='log2', bootstrap=True, class_weight='balanced', random_state=42
)

# Perform 5-fold cross-validation
folds = 5
cv_folds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
fold_accuracies = []  # Store accuracy for each fold
cv_mismatches_per_fold = {}  # Store mismatches for each fold

for fold_idx, (train_idx, val_idx) in enumerate(cv_folds.split(X_train, y_train)):
    # Split data for this fold
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Train and predict
    rf_classifier.fit(X_train_fold, y_train_fold)
    y_val_pred = rf_classifier.predict(X_val_fold)

    # Compute accuracy for this fold
    fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
    fold_accuracies.append(fold_accuracy)

    print(f"Fold {fold_idx + 1} Accuracy: {fold_accuracy:.4f}")

    # Track mismatches
    mismatches = []
    for idx, (actual, predicted) in enumerate(zip(y_val_fold, y_val_pred)):
        if actual != predicted:
            original_index = y_val_fold.index[idx]
            mismatches.append((original_index, actual, predicted))

    cv_mismatches_per_fold[f"Fold {fold_idx + 1}"] = mismatches

# Compute and print average cross-validation accuracy and standard deviation
avg_cv_accuracy = np.mean(fold_accuracies)
std_cv_accuracy = np.std(fold_accuracies)

print(f"\nAverage Cross-Validation Accuracy: {avg_cv_accuracy:.4f}")
print(f"Standard Deviation of Cross-Validation Accuracy: {std_cv_accuracy:.4f}")

# Train model on full training data
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_test_pred = rf_classifier.predict(X_test)

# Compute final test accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"\nTest Set Accuracy: {test_accuracy:.4f}")

# Find mismatches in test set
test_mismatches = []
for idx, (actual, predicted) in enumerate(zip(y_test, y_test_pred)):
    if actual != predicted:
        original_index = y_test.index[idx]
        test_mismatches.append((original_index, actual, predicted))

# Print mismatches for each fold
for fold_name, mismatches in cv_mismatches_per_fold.items():
    print(f"\n{fold_name} Mismatched Predictions:")
    if mismatches:
        for original_index, actual, predicted in mismatches:
            print(f"Index: {original_index}, Actual: {actual}, Predicted: {predicted}")
    else:
        print("No mismatches in this fold.")

# Print mismatches in test set
if test_mismatches:
    print("\nTest Set Mismatched Predictions:")
    for original_index, actual, predicted in test_mismatches:
        print(f"Index: {original_index}, Actual: {actual}, Predicted: {predicted}")
else:
    print("No mismatches in test set.")
    
    
# Predict on the test set
y_test_pred = rf_classifier.predict(X_test)

# Create a DataFrame with the predictions
predictions_df = matching_features.iloc[y_test.index][['filename', 'time', 'space']].copy()
predictions_df['actual_cluster'] = y_test.values
predictions_df['predicted_cluster'] = y_test_pred

# Save predictions to CSV
# predictions_path = "/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/rf_predictions.csv"
# predictions_df.to_csv(predictions_path, index=False)

# print(f"Predictions saved to: {predictions_path}")

