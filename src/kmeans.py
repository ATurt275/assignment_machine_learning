# kmeans.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: K-means implementation
# -------------------------------
def kmeans(X, k, max_iters=100, tol=1e-4):
    np.random.seed(42)
    # Randomly initialize centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for i in range(max_iters):
        # Compute distances and assign labels
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)

        # Compute new centroids
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tol):
            break

        centroids = new_centroids

    return labels, centroids

# -------------------------------
# Step 2: Compute WCSS
# -------------------------------
def compute_wcss(X, labels, centroids):
    return sum(np.sum((X[labels == i] - centroids[i])**2) for i in range(len(centroids)))

# -------------------------------
# Step 3: Elbow method
# -------------------------------
def elbow_method(X, k_range=range(2,9)):
    wcss_values = []
    for k in k_range:
        labels, centroids = kmeans(X, k)
        wcss_values.append(compute_wcss(X, labels, centroids))

    # Plot elbow curve
    plt.figure(figsize=(8,5))
    plt.plot(list(k_range), wcss_values, marker='o')
    plt.xlabel('Number of clusters k')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal k')
    plt.show()

    return wcss_values

# -------------------------------
# Step 4: Cluster statistics
# -------------------------------
def cluster_statistics(df, labels):
    df['cluster'] = labels
    stats = df.groupby('cluster').agg({
        'price': ['mean', 'min', 'max'],
        'units_sold': ['mean', 'min', 'max'],
        'profit': ['mean', 'min', 'max'],
        'promotion_frequency': ['mean'],
        'shelf_level': ['mean'],
        'product_id': 'count'
    })
    stats.columns = ['_'.join(col) for col in stats.columns]
    return stats

# -------------------------------
# Step 5: Cluster scatter plot
# -------------------------------
def plot_clusters_2d(X, labels, centroids, feature_x=0, feature_y=2, xlabel='Price', ylabel='Units Sold'):
    plt.figure(figsize=(8,6))
    plt.scatter(X[:, feature_x], X[:, feature_y], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centroids[:, feature_x], centroids[:, feature_y], s=300, c='red', marker='X')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('K-means Clustering')
    plt.show()
