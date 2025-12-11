# main.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import preprocessing_pipeline
from kmeans import kmeans, compute_wcss, cluster_statistics, plot_clusters_2d
from regression_models import regression_pipeline

# -------------------------
# Setup results folder
# -------------------------
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# Step 1: Preprocessing
# -------------------------
print("Running preprocessing pipeline...")
preproc_results = preprocessing_pipeline('data/product_sales.csv')
df_scaled = preproc_results['df_scaled']
df_clean = preproc_results['df_clean']

# Save cleaned and scaled data
df_clean.to_csv(os.path.join(RESULTS_DIR, 'df_clean.csv'), index=False)
df_scaled.to_csv(os.path.join(RESULTS_DIR, 'df_scaled.csv'), index=False)

print("Missing value summary:")
print(preproc_results['missing_counts'])
print(f"Number of outliers capped: {preproc_results['n_outliers']}")

# -------------------------
# Step 2: K-means Clustering
# -------------------------
print("\nRunning K-means Elbow Method...")

X = df_scaled[['price','units_sold','profit']].values  # relevant numeric features
k_range = range(2, 9)
wcss_values = []

for k in k_range:
    labels_tmp, centroids_tmp = kmeans(X, k)
    wcss_values.append(compute_wcss(X, labels_tmp, centroids_tmp))

# Save elbow plot
plt.figure(figsize=(8,5))
plt.plot(list(k_range), wcss_values, marker='o')
plt.xlabel('Number of clusters k')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.savefig(os.path.join(RESULTS_DIR, 'elbow_plot.png'))
plt.close()

# Automatically choose optimal k using maximum second derivative (curvature) method
wcss_diff = np.diff(wcss_values)
wcss_diff2 = np.diff(wcss_diff)
optimal_k = k_range[1:][np.argmin(wcss_diff2)]  # index shift
print(f"Automatically selected k = {optimal_k}")

# Run K-means with optimal k
labels, centroids = kmeans(X, optimal_k)

# Cluster statistics
stats = cluster_statistics(df_clean[['price','units_sold','profit','promotion_frequency','shelf_level','product_id']], labels)
stats.to_csv(os.path.join(RESULTS_DIR, 'cluster_statistics.csv'))
print("\nCluster Statistics:")
print(stats)

# Plot clusters
plot_clusters_2d(X, labels, centroids, feature_x=0, feature_y=1, xlabel='Price', ylabel='Units Sold')
plt.savefig(os.path.join(RESULTS_DIR, 'cluster_scatter.png'))
plt.close()

# -------------------------
# Step 3: Regression Analysis
# -------------------------
print("\nRunning regression analysis...")
reg_results = regression_pipeline(df_clean, target='profit', test_size=0.3)

# Compare models
for model_name in ['linear', 'polynomial']:
    print(f"\n{model_name.capitalize()} Regression:")
    print(f"MSE: {reg_results[model_name]['mse']:.2f}")
    print(f"MAE: {reg_results[model_name]['mae']:.2f}")

# Determine best model
best_model_name = 'linear' if reg_results['linear']['mse'] <= reg_results['polynomial']['mse'] else 'polynomial'
print(f"\nBest regression model: {best_model_name}")

# Plot Actual vs Predicted
y_test = reg_results['y_test']
y_pred = reg_results[best_model_name]['pred']

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title(f'Actual vs Predicted - {best_model_name.capitalize()} Regression')
plt.savefig(os.path.join(RESULTS_DIR, 'regression_actual_vs_predicted.png'))
plt.close()

# Bonus: Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Profit')
plt.ylabel('Residuals')
plt.title(f'Residual Plot - {best_model_name.capitalize()} Regression')
plt.savefig(os.path.join(RESULTS_DIR, 'regression_residuals.png'))
plt.close()

print("\nAll results and plots saved in 'results/' folder.")
