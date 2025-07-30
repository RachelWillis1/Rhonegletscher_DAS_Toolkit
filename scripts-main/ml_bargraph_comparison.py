#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 06:25:22 2024

@author: rachelwillis
"""

# Import necessary libraries
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Set Times New Roman as the default font
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams.update({'font.size': 23})

# Paths to uploaded files
feature_files = [
    "../data/meta_data_coherency/20200706_071820.361-20200706_091820.361/meta_data_coherency.npy",
    "../data/meta_data_coherency/20200707_070000.000-20200707_090000.000/meta_data_coherency.npy",
    "../data/meta_data_coherency/20200707_140000.000-20200707_160000.000/meta_data_coherency.npy",
    "../data/meta_data_coherency/20200708_070000.000-20200708_090000.000/meta_data_coherency.npy",
    "../data/meta_data_coherency/20200708_140000.000-20200708_160000.000/meta_data_coherency.npy",
    "../data/meta_data_coherency/20200709_070000.000-20200709_090000.000/meta_data_coherency.npy",
    "../data/meta_data_coherency/20200715_070000.000-20200715_090000.000/meta_data_coherency.npy",
    "../data/meta_data_coherency/20200715_140000.000-20200715_160000.000/meta_data_coherency.npy",
    "../data/meta_data_coherency/20200803_070000.000-20200803_090000.000/meta_data_coherency.npy"
]

clusters_path = "../data/MyClustersInfo_Expanded_CV_corrected.xlsx"

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
    feature_columns = ['space', 'time', 'filename'] + [f'feature_{i}' for i in range(features.shape[1] - 3)]
    feature_df = pd.DataFrame(features, columns=feature_columns)
    feature_df['filename'] = feature_df['filename'].astype(str).str[-44:]  # Keep last 44 characters
    feature_df['time'] = feature_df['time'].astype(float)
    feature_df['space'] = feature_df['space'].astype(int)
    combined_features.append(feature_df)

# Concatenate all feature DataFrames
all_features_df = pd.concat(combined_features, ignore_index=True)

# Filter clusters_info to exclude missing features
filtered_clusters_info = clusters_info.merge(
    all_features_df[['filename', 'time', 'space']], on=['filename', 'time', 'space'], how='inner'
)

ground_truth_labels = filtered_clusters_info['cluster']

# Filter features that match with filtered_clusters_info
matching_features = pd.merge(
    all_features_df, filtered_clusters_info[['filename', 'time', 'space']], on=['filename', 'time', 'space'], how='inner'
)

# Extract the filtered feature values
filtered_features = matching_features.drop(columns=['filename', 'time', 'space']).to_numpy()

# Initialize results storage
results = {}

# --- Unsupervised Methods ---
unsupervised_models = {
    "K-Means": KMeans(n_clusters=len(np.unique(ground_truth_labels)), init='random', n_init=10, max_iter=300, random_state=42),
    "DBSCAN": DBSCAN(eps=1, min_samples=5),
    "Agglomerative": AgglomerativeClustering(n_clusters=len(np.unique(ground_truth_labels)), linkage='single'),
    "Gaussian Mixture": GaussianMixture(n_components=len(np.unique(ground_truth_labels)), covariance_type='spherical', random_state=42)
}

for model_name, model in unsupervised_models.items():
    if model_name == "DBSCAN":
        pca = PCA(n_components=2)
        features_reduced = pca.fit_transform(filtered_features)
        model.fit(features_reduced)
        labels = model.fit_predict(features_reduced)
    else:
        model.fit(filtered_features)
        labels = model.fit_predict(filtered_features)

    acc_score = accuracy_score(ground_truth_labels, labels)
    sil_score = silhouette_score(filtered_features, labels)
    ari_score = adjusted_rand_score(ground_truth_labels, labels)
    results[model_name] = (acc_score, sil_score, ari_score)
    print(f"{model_name} - Accuracy: {acc_score:.4f}, Silhouette: {sil_score:.4f}, ARI: {ari_score:.4f}")


# --- Supervised Methods ---
X_train, X_test, y_train, y_test = train_test_split(filtered_features, ground_truth_labels, test_size=0.2, random_state=42)
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

supervised_models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=3, min_samples_split=2, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=2, max_features='log2', bootstrap=True, class_weight='balanced', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, min_samples_split=2, random_state=42),
    "Naive Bayes": GaussianNB()
}

for model_name, model in supervised_models.items():
    print(f"\n=== {model_name} ===")
    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_val_pred = model.predict(X_val_fold)
        fold_accuracies.append(accuracy_score(y_val_fold, y_val_pred))

        print(f"Fold {fold_idx + 1} Accuracy: {fold_accuracies[-1]:.4f}")

    avg_cv_accuracy = np.mean(fold_accuracies)
    var_cv_accuracy = np.var(fold_accuracies)

    model.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    results[model_name] = {"CV Accuracy": avg_cv_accuracy, "Var CV Accuracy": var_cv_accuracy, "Test Accuracy": test_accuracy}

    print(f"\nAvg CV Accuracy: {avg_cv_accuracy:.4f}")
    print(f"Test Set Accuracy: {test_accuracy:.4f}")

    # # Save the trained Random Forest model
    # if model_name == "Random Forest":
    #     joblib.dump(model, "/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/Rhone_Glacier_AWS_Final/random_forest_model_test.pkl")
    #     print("Random Forest model saved as 'random_forest_model_test.pkl'")

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'wspace': 0.05}, sharey=True)

unsupervised_methods = list(unsupervised_models.keys())
x_unsupervised = np.arange(len(unsupervised_methods))

supervised_methods = list(supervised_models.keys())
x_supervised = np.arange(len(supervised_methods))

width = 0.3

axes[0].bar(x_unsupervised, [results[m][0] for m in unsupervised_models], color="cornflowerblue", width=0.6)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Unsupervised Methods')
axes[0].set_xticks(x_unsupervised)
axes[0].set_xticklabels(unsupervised_methods, rotation=20, ha='right')
axes[1].bar(x_supervised - width/2, [results[m]["CV Accuracy"] for m in supervised_models], width, color="darkgreen")
axes[1].bar(x_supervised + width/2, [results[m]["Test Accuracy"] for m in supervised_models], width, color="lightgreen")
axes[1].set_ylim(0, 1)
axes[1].set_ylabel('')
axes[1].set_title('Supervised Methods')
axes[1].set_xticks(x_supervised)
axes[1].set_xticklabels(supervised_methods, rotation=20, ha='right')
fig.legend(["Unsupervised", "Supervised Cross Validation", "Supervised Test"], loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
# plt.savefig('ml_bargraph.pdf', dpi=300, bbox_inches="tight", format='pdf')
plt.show()
