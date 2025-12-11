# src/visualizations.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_actual_vs_predicted(df: pd.DataFrame, actual_col='Actual_Profit',
                             pred_col='Linear_Pred', title='Actual vs Predicted',
                             save_path=None):
    """
    Scatter plot of actual vs predicted values.
    """
    plt.figure(figsize=(8,6))
    plt.scatter(df[actual_col], df[pred_col], alpha=0.7, label='Predictions')
    # Diagonal line (perfect prediction)
    min_val = min(df[actual_col].min(), df[pred_col].min())
    max_val = max(df[actual_col].max(), df[pred_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved: {save_path}")
    plt.show()

def plot_residuals(df: pd.DataFrame, actual_col='Actual_Profit', pred_col='Linear_Pred',
                   title='Residual Plot', save_path=None):
    """
    Residual plot: Predicted vs Residuals.
    """
    residuals = df[pred_col] - df[actual_col]
    plt.figure(figsize=(8,6))
    plt.scatter(df[pred_col], residuals, alpha=0.7)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved: {save_path}")
    plt.show()

def compare_models(df: pd.DataFrame, actual_col='Actual_Profit', model_cols=['Linear_Pred','Polynomial_Pred'],
                   title='Model Comparison', save_path=None):
    """
    Overlay actual vs predicted for multiple models.
    """
    plt.figure(figsize=(8,6))
    plt.scatter(df[actual_col], df[actual_col], c='k', alpha=0.5, label='Actual')  # perfect line
    colors = ['b','g','orange','purple']
    for i, col in enumerate(model_cols):
        plt.scatter(df[actual_col], df[col], alpha=0.7, color=colors[i % len(colors)], label=col)
    plt.plot([df[actual_col].min(), df[actual_col].max()],
             [df[actual_col].min(), df[actual_col].max()],
             'r--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved: {save_path}")
    plt.show()

# --- Example usage when run as script ---
if __name__ == "__main__":
    csv_path = 'results/regression_predictions.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run regression first.")
    
    df = pd.read_csv(csv_path)

    # Linear regression plots
    plot_actual_vs_predicted(df, pred_col='Linear_Pred', title='Linear Regression: Actual vs Predicted',
                            save_path='results/linear_actual_vs_predicted.png')
    plot_residuals(df, pred_col='Linear_Pred', title='Linear Regression Residuals',
                   save_path='results/linear_residuals.png')

    # Polynomial regression plots
    plot_actual_vs_predicted(df, pred_col='Polynomial_Pred', title='Polynomial Regression: Actual vs Predicted',
                            save_path='results/poly_actual_vs_predicted.png')
    plot_residuals(df, pred_col='Polynomial_Pred', title='Polynomial Regression Residuals',
                   save_path='results/poly_residuals.png')

    # Comparison overlay
    compare_models(df, model_cols=['Linear_Pred','Polynomial_Pred'], title='Regression Model Comparison',
                   save_path='results/model_comparison.png')
