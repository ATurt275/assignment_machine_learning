# src/run_regression.py

import pandas as pd
from regression_models import regression_pipeline
import os

# -------------------------------
# 1️⃣ Load preprocessed & scaled data
# -------------------------------
data_path = 'data/product_sales_scaled.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found. Run preprocessing.py first!")

df_scaled = pd.read_csv(data_path)

# -------------------------------
# 2️⃣ Run regression pipeline
# -------------------------------
results = regression_pipeline(df_scaled, target='profit')

# -------------------------------
# 3️⃣ Print evaluation metrics
# -------------------------------
print("\n=== Regression Model Performance ===")
for model_name, metrics in results.items():
    if model_name in ['linear', 'polynomial']:
        print(f"{model_name.title()} Regression -> MSE: {metrics['mse']:.2f}, MAE: {metrics['mae']:.2f}")

# -------------------------------
# 4️⃣ Save predictions for visualization
# -------------------------------
output_df = results['X_test'].copy()
output_df['Actual_Profit'] = results['y_test'].values
output_df['Linear_Pred'] = results['linear']['pred']
output_df['Polynomial_Pred'] = results['polynomial']['pred']

os.makedirs('results', exist_ok=True)
output_csv_path = 'results/regression_predictions.csv'
output_df.to_csv(output_csv_path, index=False)
print(f"\nPredictions saved to {output_csv_path}")
