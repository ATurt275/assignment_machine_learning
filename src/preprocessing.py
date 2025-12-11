# src/preprocessing.py
"""
Preprocessing utilities for the Product Sales ML project.
"""
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import pathlib

# --- Load Data ---
def load_data(path: str = 'data/product_sales.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

# --- Missing Value Report ---
def missing_value_report(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    counts = df.isna().sum()
    frac = df.isna().mean()
    return counts, frac

# --- Handle Missing Values ---
def handle_missing_values(
    df: pd.DataFrame,
    strategy_numeric: str = 'median',
    drop_threshold: float = 0.5,
    numeric_columns: List[str] = None,
    categorical_columns: List[str] = None
) -> pd.DataFrame:
    df = df.copy()
    row_missing_frac = df.isna().mean(axis=1)
    df = df[row_missing_frac <= drop_threshold].reset_index(drop=True)

    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if numeric_columns:
        num_imp = SimpleImputer(strategy=strategy_numeric)
        df[numeric_columns] = num_imp.fit_transform(df[numeric_columns])

    for c in categorical_columns:
        if df[c].isna().any():
            mode_vals = df[c].mode()
            mode_val = mode_vals[0] if not mode_vals.empty else 'Unknown'
            df[c] = df[c].fillna(mode_val)

    # restore integer-like numeric columns
    for c in numeric_columns:
        if np.all(np.isfinite(df[c])) and (df[c] == df[c].astype(int)).all():
            df[c] = df[c].astype(int)
    return df

# --- Outlier Detection ---
def detect_outliers_iqr(df: pd.DataFrame, columns: List[str], k: float = 1.5) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        mask |= (df[col] < lower) | (df[col] > upper)
    return mask

def cap_outliers(df: pd.DataFrame, columns: List[str], k: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df

# --- Scaling ---
def scale_features(df: pd.DataFrame, columns: List[str], method: str = 'zscore'):
    df_scaled = df.copy()
    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'zscore' or 'minmax'")
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled, scaler

# --- Full Pipeline ---
def preprocessing_pipeline(path: str = 'data/product_sales.csv'):
    df_raw = load_data(path)
    missing_counts, missing_frac = missing_value_report(df_raw)

    df_clean = handle_missing_values(df_raw, strategy_numeric='median', drop_threshold=0.5)
    numeric_cols = [c for c in ['price','cost','units_sold','promotion_frequency','shelf_level','profit'] if c in df_clean.columns]
    outlier_mask = detect_outliers_iqr(df_clean, numeric_cols)
    n_outliers = outlier_mask.sum()
    df_capped = cap_outliers(df_clean, numeric_cols)
    df_scaled, scaler = scale_features(df_capped, numeric_cols, method='zscore')

    return {
        'df_raw': df_raw,
        'df_clean': df_clean,
        'df_capped': df_capped,
        'df_scaled': df_scaled,
        'scaler': scaler,
        'missing_counts': missing_counts,
        'missing_frac': missing_frac,
        'n_outliers': n_outliers
    }
