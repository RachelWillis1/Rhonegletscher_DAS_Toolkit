#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 06:25:22 2024

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


## . . Unsupervised Methods

# Unsupervised: K-Means Manual Search
best_kmeans_score = 0
best_kmeans_silhouette = 0
best_kmeans_ari = 0
for init in ['k-means++', 'random']:
    for n_init in [10, 20]:
        for max_iter in [300, 600]:
            kmeans = KMeans(n_clusters=len(np.unique(ground_truth_labels)), init=init, n_init=n_init, max_iter=max_iter, random_state=42)
            kmeans_labels = kmeans.fit_predict(filtered_features)
            acc_score = accuracy_score(ground_truth_labels, kmeans_labels)
            sil_score = silhouette_score(filtered_features, kmeans_labels)
            ari_score = adjusted_rand_score(ground_truth_labels, kmeans_labels)
            if acc_score > best_kmeans_score:
                best_kmeans_score = acc_score
                best_kmeans_silhouette = sil_score
                best_kmeans_ari = ari_score
                best_params = {"init": init, "n_init": n_init, "max_iter": max_iter}
results['K-Means'] = (best_kmeans_score, best_kmeans_silhouette, best_kmeans_ari)
print(f"\n K-Means - Accuracy: {best_kmeans_score:.4f}, Silhouette: {best_kmeans_silhouette:.4f}, ARI: {best_kmeans_ari:.4f}")
print(f" Best Parameters: init={best_params['init']}, n_init={best_params['n_init']}, max_iter={best_params['max_iter']}")

pca = PCA(n_components=2)
features_reduced = pca.fit_transform(filtered_features)

# Unsupervised: DBSCAN Manual Search
best_dbscan_score = 0
best_dbscan_silhouette = 0
best_dbscan_ari = 0
for eps in [1, 5, 10, 50, 100]:
    for min_samples in [3, 5, 10]:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(features_reduced)
        # Skip cases where all points are in a single cluster
        if len(np.unique(dbscan_labels)) <= 1:
            continue

        try:
            acc_score = accuracy_score(ground_truth_labels, dbscan_labels)
            sil_score = silhouette_score(filtered_features, dbscan_labels)
            ari_score = adjusted_rand_score(ground_truth_labels, dbscan_labels)
        except ValueError as e:
            print(f"Skipping due to error: {e}")
            continue

        if acc_score > best_dbscan_score:
            best_dbscan_score = acc_score
            best_dbscan_silhouette = sil_score
            best_dbscan_ari = ari_score
            best_params = {"eps": eps, "min_samples": min_samples}
results['DBSCAN'] = (best_dbscan_score, best_dbscan_silhouette, best_dbscan_ari)
print(f"\n DBSCAN - Accuracy: {best_dbscan_score:.4f}, Silhouette: {best_dbscan_silhouette:.4f}, ARI: {best_dbscan_ari:.4f}")
print(f" Best Parameters: eps={best_params['eps']}, min_samples={best_params['min_samples']}")

# Unsupervised: Agglomerative Clustering Manual Search
best_agglo_score = 0
best_agglo_silhouette = 0
best_agglo_ari = 0
for linkage in ['ward', 'complete', 'average', 'single']:
    agglo = AgglomerativeClustering(n_clusters=len(np.unique(ground_truth_labels)), linkage=linkage)
    agglo_labels = agglo.fit_predict(filtered_features)
    acc_score = accuracy_score(ground_truth_labels, agglo_labels)
    sil_score = silhouette_score(filtered_features, agglo_labels)
    ari_score = adjusted_rand_score(ground_truth_labels, agglo_labels)
    if acc_score > best_agglo_score:
        best_agglo_score = acc_score
        best_agglo_silhouette = sil_score
        best_agglo_ari = ari_score
        best_params = {"linkage": linkage}
results['Agglomerative'] = (best_agglo_score, best_agglo_silhouette, best_agglo_ari)
print(f"\n Agglomerative - Accuracy: {best_agglo_score:.4f}, Silhouette: {best_agglo_silhouette:.4f}, ARI: {best_agglo_ari:.4f}")
print(f" Best Parameters: linkage={best_params['linkage']}")

# Unsupervised: Gaussian Mixture Manual Search
best_gmm_score = 0
best_gmm_silhouette = 0
best_gmm_ari = 0
for cov_type in ['full', 'tied', 'diag', 'spherical']:
    gmm = GaussianMixture(n_components=len(np.unique(ground_truth_labels)), covariance_type=cov_type, random_state=42)
    gmm_labels = gmm.fit_predict(filtered_features)
    acc_score = accuracy_score(ground_truth_labels, gmm_labels)
    sil_score = silhouette_score(filtered_features, gmm_labels)
    ari_score = adjusted_rand_score(ground_truth_labels, gmm_labels)
    if acc_score > best_gmm_score:
        best_gmm_score = acc_score
        best_gmm_silhouette = sil_score
        best_gmm_ari = ari_score
        best_params = {"cov_type": cov_type}

results['Gaussian Mixture'] = (best_gmm_score, best_gmm_silhouette, best_gmm_ari)
print(f"\n Gaussian Mixture - Accuracy: {best_gmm_score:.4f}, Silhouette: {best_gmm_silhouette:.4f}, ARI: {best_gmm_ari:.4f}")
print(f" Best Parameters: linkage={best_params['cov_type']}")


## . . Supervised Methods

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(filtered_features, ground_truth_labels, test_size=0.2, random_state=42)

# Cross-validation folds
cv_folds = 5 #StratifiedKFold(n_splits=5, shuffle=True, random_state=42)#5
# cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

# Supervised: Decision Tree Grid Search with Cross-Validation
# Hyperparameter grid for Decision Tree
param_grid = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=2)
grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
# Compute mean cross-validation accuracy
cv_accuracies = grid_search.cv_results_['mean_test_score']
mean_cv_accuracy = np.mean(cv_accuracies)
print(f"\n Mean DT Cross-Validation Accuracy: {mean_cv_accuracy:.4f}")
print(f" Best DT Parameters: {grid_search.best_params_}")

# Test the final model on the test set
final_model = grid_search.best_estimator_
y_test_pred = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')
results['Decision Tree'] = (grid_search.best_score_, test_accuracy)
print(f" Test set accuracy with best dt model: {test_accuracy:.4f}")
print(f" Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")



# Supervised: Random Forest Grid Search with Cross-Validation
# Hyperparameter grid for Random Forest
param_grid = {'n_estimators': [100, 200, 300],'max_depth': [10, 20, None],
              'min_samples_split': [2, 5, 10],'max_features': ['sqrt', 'log2', None],
              'bootstrap': [True, False], 'class_weight': ['balanced', 'balanced_subsample']}

rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
# Compute mean cross-validation accuracy
cv_accuracies = grid_search.cv_results_['mean_test_score']
mean_cv_accuracy = np.mean(cv_accuracies)
print(f"\n Mean RF Cross-Validation Accuracy: {mean_cv_accuracy:.4f}")
print(f" Best RF Parameters: {grid_search.best_params_}")

# Test the final model on the test set
final_model = grid_search.best_estimator_
y_test_pred = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')
results['Random Forest'] = (grid_search.best_score_, test_accuracy)
print(f" Test set accuracy with best rf model: {test_accuracy:.4f}")
print(f" Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")




# Supervised: Gradient Boosting Grid Search with Cross-Validation
# Hyperparameter grid for Gradient Boosting
param_grid = {'n_estimators': [50, 100, 200],'learning_rate': [0.01, 0.1, 0.2],
              'max_depth': [3, 5, 10]}
gb_classifier = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=5)
grid_search = GridSearchCV(estimator=gb_classifier, param_grid=param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
# Compute mean cross-validation accuracy
cv_accuracies = grid_search.cv_results_['mean_test_score']
mean_cv_accuracy = np.mean(cv_accuracies)
print(f"\n Mean GB Cross-Validation Accuracy: {mean_cv_accuracy:.4f}")
print(f" Best GB Parameters: {grid_search.best_params_}")

# Test the final model on the test set
final_model = grid_search.best_estimator_
y_test_pred = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')
results['Gradient Boosting'] = (grid_search.best_score_, test_accuracy)
print(f" Test set accuracy with best gb model: {test_accuracy:.4f}")
print(f" Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")



# Supervised: Naive Bayes with Cross-Validation
nb_classifier = GaussianNB()
scores = cross_val_score(nb_classifier, X_train, y_train, cv=cv_folds, scoring='accuracy')
print(f"\n Mean NB Cross-Validation Accuracy: {scores.mean():.4f}")

# Test the final model on the test set for Naive Bayes
nb_classifier.fit(X_train, y_train)  # Train NB on full training set
y_test_pred = nb_classifier.predict(X_test)  # Predict with NB
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')
results['Naive Bayes'] = (scores.mean(), test_accuracy)
print(f" Test set accuracy with NB: {test_accuracy:.4f}")
print(f" Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
