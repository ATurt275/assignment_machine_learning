# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os

# Import your modules (assumes they are in src/ and importable)
from preprocessing import preprocessing_pipeline
from kmeans import kmeans, compute_wcss, cluster_statistics
from regression_models import regression_pipeline

st.set_page_config(page_title="Product Performance Analysis", layout="wide")

# -------------------------
# Helper functions
# -------------------------
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def plot_elbow(k_range, wcss_values):
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.plot(list(k_range), wcss_values, marker='o')
    ax.set_xlabel('Number of clusters k')
    ax.set_ylabel('WCSS')
    ax.set_title('Elbow Method for Optimal k')
    ax.grid(True)
    return fig

def plot_clusters_2d_from_arrays(X, labels, centroids, x_idx=0, y_idx=1, xlabel='Feature X', ylabel='Feature Y'):
    fig, ax = plt.subplots(figsize=(7,5))
    scatter = ax.scatter(X[:, x_idx], X[:, y_idx], c=labels, cmap='tab10', alpha=0.7)
    if centroids is not None:
        ax.scatter(centroids[:, x_idx], centroids[:, y_idx], s=200, c='red', marker='X', label='Centroids')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title('K-means Clusters (2D view)')
    return fig

def plot_actual_vs_pred(y_true, y_pred, title='Actual vs Predicted'):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_true, y_pred, alpha=0.7)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.6)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    ax.grid(True)
    return fig

def plot_residuals(y_true, y_pred, title='Residuals'):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(y_pred, residuals, alpha=0.7)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residual (Actual - Predicted)')
    ax.set_title(title)
    ax.grid(True)
    return fig

# -------------------------
# App UI
# -------------------------
st.title("📊 Product Performance Analysis")
st.write("Interactive presentation of preprocessing, K-means clustering, and regression results.")

# Sidebar controls
st.sidebar.header("Configuration")
run_preproc = st.sidebar.button("Run preprocessing & load data")
k_range = st.sidebar.slider("Elbow k range (max)", 4, 12, value=8)
selected_features = st.sidebar.multiselect(
    "Features to use for clustering (numeric)",
    options=['price','cost','units_sold','promotion_frequency','shelf_level','profit'],
    default=['price','units_sold','profit']
)
auto_k = st.sidebar.checkbox("Auto-select k (elbow curvature)", value=True)
chosen_k = st.sidebar.slider("Manual k (if not auto)", 2, 12, value=3)
reg_target = st.sidebar.selectbox("Regression target", options=['profit','units_sold'], index=0)
poly_degree = st.sidebar.slider("Polynomial degree (for poly model)", 2, 3, value=2)

# Run preprocessing immediately (also run when button pressed)
if 'preproc_results' not in st.session_state or run_preproc:
    st.session_state['preproc_results'] = preprocessing_pipeline('data/product_sales.csv')
results = st.session_state['preproc_results']

df_raw = results['df_raw']
df_clean = results['df_clean']
df_capped = results['df_capped']
df_scaled = results['df_scaled']

# Data Overview
st.header("1. Data Overview")
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Raw data (sample)")
    st.dataframe(df_raw.head(8))
with col2:
    st.subheader("Preprocessing summary")
    st.write("Missing values per column:")
    st.write(results['missing_counts'])
    st.write(f"Number of outliers detected (any numeric): **{int(results['n_outliers'])}**")

# Save cleaned/scaled downloads
st.download_button(
    label="📥 Download cleaned CSV",
    data=dataframe_to_csv_bytes(df_clean),
    file_name="product_sales_clean.csv",
    mime="text/csv"
)
st.download_button(
    label="📥 Download scaled CSV",
    data=dataframe_to_csv_bytes(df_scaled),
    file_name="product_sales_scaled.csv",
    mime="text/csv"
)

st.markdown("""
**Preprocessing actions:**  
- Missing values: median (numeric), mode (categorical).  
- Outliers: detected by IQR and capped (winsorized) to reduce distortion.  
- Scaling: Z-score standardization for numeric features (mean=0, std=1).
""")

# -------------------------
# K-means Clustering Section
# -------------------------
st.header("2. K-means Clustering")

if len(selected_features) < 2:
    st.warning("Select at least 2 features for clustering.")
else:
    X = df_scaled[selected_features].to_numpy()

    # Compute WCSS for elbow
    k_values = range(2, k_range+1)
    wcss_vals = []
    for kk in k_values:
        labels_tmp, cents_tmp = kmeans(X, kk)
        wcss_vals.append(compute_wcss(X, labels_tmp, cents_tmp))

    fig_elbow = plot_elbow(k_values, wcss_vals)
    st.pyplot(fig_elbow)

    # Auto-select k using curvature (second derivative) if requested
    if auto_k and len(wcss_vals) >= 3:
        dif1 = np.diff(wcss_vals)
        dif2 = np.diff(dif1)
        # choose the k with the largest drop in second derivative (most negative dif2)
        idx = np.argmin(dif2) if len(dif2)>0 else 0
        auto_selected_k = list(k_values)[1:][idx]  # shift because dif2 length = len(k_values)-2
        st.write(f"Auto-selected k (elbow curvature): **{auto_selected_k}**")
        k_use = auto_selected_k
    else:
        k_use = chosen_k
        st.write(f"Using manual k = **{k_use}**")

    # Run K-means with k_use
    labels, cents = kmeans(X, k_use)
    df_clusters = df_scaled.copy()
    df_clusters['cluster'] = labels

    # Show cluster statistics (use columns from df_clean for interpretable numbers)
    display_df_for_stats = df_clean[['product_id','price','units_sold','profit','promotion_frequency','shelf_level']].copy()
    stats = cluster_statistics(display_df_for_stats, labels)
    st.subheader("Cluster statistics")
    st.dataframe(stats)

    # Allow download of cluster-labeled data
    st.download_button(
        label="📥 Download cluster assignments (CSV)",
        data=dataframe_to_csv_bytes(df_clusters),
        file_name="clustered_products.csv",
        mime="text/csv"
    )

    # 2D scatter plot: allow user to pick axes
    st.subheader("Cluster scatter (2D)")
    x_axis = st.selectbox("X axis", options=selected_features, index=0)
    y_axis = st.selectbox("Y axis", options=selected_features, index=1 if len(selected_features)>1 else 0)
    x_idx = selected_features.index(x_axis)
    y_idx = selected_features.index(y_axis)
    fig_clusters = plot_clusters_2d_from_arrays(X, labels, cents, x_idx=x_idx, y_idx=y_idx,
                                               xlabel=x_axis, ylabel=y_axis)
    st.pyplot(fig_clusters)
    # Save to results/ folder
    save_path = os.path.join("results", f"clusters_{x_axis}_vs_{y_axis}_k{k_use}.png")
    fig_clusters.savefig(save_path, bbox_inches='tight')
    st.write(f"Saved cluster plot to `{save_path}`")

# -------------------------
# Regression Section
# -------------------------
st.header("3. Regression Analysis (Predicting target)")

st.write(f"**Selected target:** `{reg_target}`")
# Use df_scaled for features (scaled), keep original unscaled target if preferable.
# We'll create a regression input DF with scaled features and original target (so predictions are in original units).
reg_df = df_scaled.copy()
# If target is in df_clean and not scaled, replace with df_clean[target] for interpretable predictions
if reg_target in df_clean.columns:
    reg_df[reg_target] = df_clean[reg_target].values

if st.button("Run regression pipeline"):
    reg_results = regression_pipeline(reg_df, target=reg_target, test_size=0.3)

    # Show evaluation metrics
    st.subheader("Model performance")
    metrics_table = pd.DataFrame({
        'model': ['Linear', 'Polynomial'],
        'mse': [reg_results['linear']['mse'], reg_results['polynomial']['mse']],
        'mae': [reg_results['linear']['mae'], reg_results['polynomial']['mae']]
    })
    st.table(metrics_table)

    # Determine best model
    best_model = 'linear' if reg_results['linear']['mse'] <= reg_results['polynomial']['mse'] else 'polynomial'
    st.write(f"**Best model (by MSE):** {best_model}")

    # Create predictions dataframe for download / viewing
    preds_df = reg_results['X_test'].copy()
    preds_df['actual'] = reg_results['y_test'].values
    preds_df['pred_linear'] = reg_results['linear']['pred']
    preds_df['pred_poly'] = reg_results['polynomial']['pred']

    st.subheader("Predictions (test set sample)")
    st.dataframe(preds_df.head(12))

    st.download_button(
        label="📥 Download regression predictions CSV",
        data=dataframe_to_csv_bytes(preds_df),
        file_name="regression_predictions.csv",
        mime="text/csv"
    )

    # Plots: Actual vs Predicted & Residuals for both models
    st.subheader("Actual vs Predicted - Linear")
    fig_lin = plot_actual_vs_pred(reg_results['y_test'].values, reg_results['linear']['pred'], title='Linear: Actual vs Predicted')
    st.pyplot(fig_lin)
    fig_lin.savefig(os.path.join("results", "linear_actual_vs_pred.png"), bbox_inches='tight')

    st.subheader("Residuals - Linear")
    fig_lin_res = plot_residuals(reg_results['y_test'].values, reg_results['linear']['pred'], title='Linear Residuals')
    st.pyplot(fig_lin_res)
    fig_lin_res.savefig(os.path.join("results", "linear_residuals.png"), bbox_inches='tight')

    st.subheader("Actual vs Predicted - Polynomial")
    fig_poly = plot_actual_vs_pred(reg_results['y_test'].values, reg_results['polynomial']['pred'], title='Poly: Actual vs Predicted')
    st.pyplot(fig_poly)
    fig_poly.savefig(os.path.join("results", "poly_actual_vs_pred.png"), bbox_inches='tight')

    st.subheader("Residuals - Polynomial")
    fig_poly_res = plot_residuals(reg_results['y_test'].values, reg_results['polynomial']['pred'], title='Poly Residuals')
    st.pyplot(fig_poly_res)
    fig_poly_res.savefig(os.path.join("results", "poly_residuals.png"), bbox_inches='tight')

    st.success("Regression run complete — plots saved to `results/`.")

else:
    st.info("Click **Run regression pipeline** to train models and view results.")

# -------------------------
# Footer / Notes
# -------------------------
st.markdown("---")
st.write("**Notes:**")
st.write("- Use the sidebar controls to adjust k-range, features, and regression target.")
st.write("- All saved plot files are written to the `results/` folder.")
st.write("- The app uses the preprocessing decisions coded in `preprocessing.py` (median imputation, IQR capping, z-score scaling).")
