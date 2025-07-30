#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:11:14 2024

@author: rachelwillis
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import skfuzzy as fuzz
import joblib


# Load your data
coh = np.load('/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/Rhone_Glacier_AWS_Final/Rhone_Glacier_STA_LTA_Long_Run2/codes/output/extr_features/coh_temp.npy')
velo = np.load('/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/Rhone_Glacier_AWS_Final/Rhone_Glacier_STA_LTA_Long_Run2/codes/output/extr_features/velo_temp.npy')

X = coh #np.hstack((coh, velo))

###############################################################################
# Elbow and Silhouette Tests

# Define the range of clusters for evaluation
cluster_range = range(2, 11)  # You can adjust this based on your needs
inertia_kmeans = []
silhouette_scores_kmeans = []
silhouette_scores_agglomerative = []
silhouette_scores_fcm = []
agg_avg_distances = []
fcm_objective = []

# Apply KMeans and Agglomerative clustering
for n_clusters in cluster_range:
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    inertia_kmeans.append(kmeans.inertia_)  # Inertia for Elbow Method
    labels_kmeans = kmeans.labels_
    silhouette_scores_kmeans.append(silhouette_score(X, labels_kmeans))  # Silhouette Score for KMeans

    # Agglomerative Clustering
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
    agg_cluster.fit_predict(X)
    labels_agg = agg_cluster.labels_
    silhouette_scores_agglomerative.append(silhouette_score(X, labels_agg))  # Silhouette Score for Agglomerative
    
    # Fuzzy C-Means
    cntr, u, _, _, _, _, _ = fuzz.cmeans(X.T, c=n_clusters, m=2, error=0.005, maxiter=1000)
    labels_fcm = np.argmax(u, axis=0)  # Assign each point to the cluster with the highest membership
    silhouette_scores_fcm.append(silhouette_score(X, labels_fcm))  # Silhouette Score for Fuzzy C-Means
    
    # Fuzzy C-Means Objective Function Calculation
    objective_value = 0
    for i in range(n_clusters):
        for j in range(X.shape[0]):
            # Calculate the squared distance between data point and centroid
            distance = np.linalg.norm(X[j] - cntr[i])
            # Weighted by the membership value
            objective_value += (u[i, j] ** 2) * distance ** 2
    fcm_objective.append(objective_value)

    # Calculate average distance within each cluster for Agglomerative
    avg_distance = 0
    for i in range(n_clusters):
        cluster_points = X[labels_agg == i]
        if len(cluster_points) > 1:
            # Calculate pairwise distances within the cluster and take the mean
            avg_distance += np.mean(pairwise_distances(cluster_points))  # Average distance
    agg_avg_distances.append(avg_distance / n_clusters)  # Store average distance

# Plot the Elbow and Silhouette Method results
plt.figure(figsize=(28, 5))

# Subplot for KMeans
plt.subplot(1, 4, 1)
plt.plot(cluster_range, inertia_kmeans, marker='o', color='red')
plt.title('Elbow Method for K-Means')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(cluster_range)

# Subplot for Agglomerative Elbow Method
plt.subplot(1, 4, 2)
plt.plot(cluster_range, agg_avg_distances, marker='o', color='blue')
plt.title('Elbow Method for Agglomerative Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Distance')
plt.xticks(cluster_range)

# Subplot for Fuzzy C-Means Elbow Method
plt.subplot(1, 4, 3)
plt.plot(cluster_range, fcm_objective, marker='o', color='purple')
plt.title('Elbow Method for Fuzzy C-Means')
plt.xlabel('Number of Clusters')
plt.ylabel('Objective Function Value')
plt.xticks(cluster_range)

# Subplot for Silhouette Method
plt.subplot(1, 4, 4)
plt.plot(cluster_range, silhouette_scores_kmeans, marker='o', color='orange', label='KMeans')
plt.plot(cluster_range, silhouette_scores_agglomerative, marker='o', color='green', label='Agglomerative')
plt.plot(cluster_range, silhouette_scores_fcm, marker='o', color='purple', label='Fuzzy C-Means')  # FCM silhouette
plt.title('Silhouette Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(cluster_range)
plt.legend()

# Show all plots
plt.tight_layout()
plt.show()

###############################################################################
# PCA Visualizations

n_clusters = 7

# KMeans visualization
kmeans_optimal = KMeans(n_clusters = n_clusters, random_state=0).fit(X)

# Reduce dimensionality to 2 components using PCA for KMeans & Agglomerative
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Get cluster labels and centroids (project centroids into PCA space)
labels_kmeans = kmeans_optimal.labels_
centroids_kmeans = kmeans_optimal.cluster_centers_
centroids_pca_kmeans = pca.transform(centroids_kmeans)

# np.save('/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/Rhone_Glacier_AWS_Final/Rhone_Glacier_STA_LTA_Long_Run2/codes/output/coh_kmeans_stalta_4clusters.npy', labels_kmeans)
# joblib.dump(kmeans_optimal, '/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/Rhone_Glacier_AWS_Final/Rhone_Glacier_STA_LTA_Long_Run2/codes/output/coh_kmeans_stalta_4clusters_model.pkl')

# Create the scatter plot for KMeans
plt.figure(figsize=(10, 6))

# Use Viridis colormap from matplotlib
cmap = plt.cm.viridis

# Set discrete boundaries for the colorbar
boundaries = np.linspace(0, n_clusters, n_clusters + 1)
norm = mcolors.BoundaryNorm(boundaries, cmap.N)

scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap=cmap, marker='o', edgecolor='k', s=50, norm=norm)
plt.scatter(centroids_pca_kmeans[:, 0], centroids_pca_kmeans[:, 1], c='red', marker='x', s=100, label="Centroids")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(labels_kmeans)  # Set the array for ScalarMappable

# Create the colorbar with discrete ticks at boundaries
cbar = plt.colorbar(sm, ticks=np.arange(0, n_clusters + 1))  # Discrete ticks

# Adjust tick labels to be in the middle of the segments
midpoints = (boundaries[:-1] + boundaries[1:]) / 2  # Calculate midpoints of each segment
cbar.set_ticks(midpoints)  # Set the ticks to the midpoints
cbar.ax.set_yticklabels([f'{i}' for i in np.arange(0, n_clusters + 1)])  # Set the labels
cbar.set_label('Clusters')

plt.title('PCA Projection of K-Means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Agglomerative Clustering visualization
agg_cluster_optimal = AgglomerativeClustering(n_clusters = n_clusters).fit(X)

# Find cluster labels
labels_agglomerative = agg_cluster_optimal.labels_
centroids_agglomerative = np.array([X[labels_agglomerative == i].mean(axis=0) for i in np.unique(labels_agglomerative)])
centroids_pca_agglomerative = pca.transform(centroids_agglomerative)

np.save('/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/Rhone_Glacier_AWS_Final/Rhone_Glacier_STA_LTA_Long_Run2/codes/output/coh_agg_stalta_7clusters.npy', labels_agglomerative)
joblib.dump(agg_cluster_optimal, '/Users/rachelwillis/myDocuments/CSMresearch/Rhone_Glacier/Rhone_Glacier_AWS_Final/Rhone_Glacier_STA_LTA_Long_Run2/codes/output/coh_agg_stalta_7clusters_model.pkl')

# Create the scatter plot for Agglomerative Clustering
plt.figure(figsize=(10, 6))

# Use Viridis colormap from matplotlib
cmap = plt.cm.viridis

# Set discrete boundaries for the colorbar
boundaries = np.linspace(0, n_clusters, n_clusters + 1)
norm = mcolors.BoundaryNorm(boundaries, cmap.N)

scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_agglomerative, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.scatter(centroids_pca_agglomerative[:, 0], centroids_pca_agglomerative[:, 1], color='red', marker='x', s=100, label='Centroids')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(labels_agglomerative)  # Set the array for ScalarMappable

# Create the colorbar with discrete ticks at boundaries
cbar = plt.colorbar(sm, ticks=np.arange(0, n_clusters + 1))  # Discrete ticks

# Adjust tick labels to be in the middle of the segments
midpoints = (boundaries[:-1] + boundaries[1:]) / 2  # Calculate midpoints of each segment
cbar.set_ticks(midpoints)  # Set the ticks to the midpoints
cbar.ax.set_yticklabels([f'{i}' for i in np.arange(0, n_clusters + 1)])
cbar.set_label('Clusters')

plt.title('PCA Projection of Agglomerative Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()


# Fuzzy C-Means vizualization
cntr, u, _, _, _, _, _ = fuzz.cmeans(X.T, c=n_clusters, m=2, error=0.005, maxiter=1000)

# u is the membership matrix
labels_fcm = np.argmax(u, axis=0)  # Assign each point to the cluster with the highest membership
centroids_fcm_pca = pca.transform(cntr)  # This transforms the centroids into the PCA space

# Plot Fuzzy C-Means results
plt.figure(figsize=(10, 6))

# Use Viridis colormap from matplotlib
cmap = plt.cm.viridis

# Set discrete boundaries for the colorbar
boundaries = np.linspace(0, n_clusters, n_clusters + 1)
norm = mcolors.BoundaryNorm(boundaries, cmap.N)

# Scatter plot of the data with fuzzy clustering results
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_fcm, cmap=cmap, marker='o', edgecolor='k', s=50)
plt.scatter(centroids_fcm_pca[:, 0], centroids_fcm_pca[:, 1], color='red', marker='x', s=100, label='Centroids')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(labels_agglomerative)  # Set the array for ScalarMappable

# Create the colorbar with discrete ticks at boundaries
cbar = plt.colorbar(sm, ticks=np.arange(0, n_clusters + 1))  # Discrete ticks

# Adjust tick labels to be in the middle of the segments
midpoints = (boundaries[:-1] + boundaries[1:]) / 2  # Calculate midpoints of each segment
cbar.set_ticks(midpoints)  # Set the ticks to the midpoints
cbar.ax.set_yticklabels([f'{i}' for i in np.arange(0, n_clusters + 1)])
cbar.set_label('Clusters')

plt.title('PCA Projection of Fuzzy C-Means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

