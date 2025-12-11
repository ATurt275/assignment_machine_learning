# run_kmeans.py

import pandas as pd
from kmeans import kmeans, elbow_method, cluster_statistics, plot_clusters_2d

# -------------------------------
# 1️⃣ Load preprocessed & scaled data
# -------------------------------
df_scaled = pd.read_csv('data/product_sales_scaled.csv')

# -------------------------------
# 2️⃣ Select numeric features for clustering
# -------------------------------
features = ['price','cost','units_sold','promotion_frequency','shelf_level','profit']
X = df_scaled[features].values

# -------------------------------
# 3️⃣ Elbow method to find optimal k
# -------------------------------
print("Running elbow method to determine optimal k...")
wcss = elbow_method(X)  # This will display the elbow plot

# -------------------------------
# 4️⃣ Run K-means with chosen k
# -------------------------------
# You can change k based on elbow plot (example: k=3)
k = 3
labels, centroids = kmeans(X, k)
print(f"K-means clustering completed with k={k}.")

# -------------------------------
# 5️⃣ Generate cluster statistics
# -------------------------------
stats = cluster_statistics(df_scaled.copy(), labels)
print("\nCluster Statistics:")
print(stats)

# -------------------------------
# 6️⃣ Plot clusters (2D visualization)
# -------------------------------
# Example: feature_x=0 (price), feature_y=2 (units_sold)
plot_clusters_2d(X, labels, centroids, feature_x=0, feature_y=2, xlabel='Price', ylabel='Units Sold')
