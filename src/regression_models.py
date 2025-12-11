# regression_models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_polynomial_regression(X_train, y_train, degree=2):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    return model, poly

def evaluate_model(model, X_test, y_test, poly=None):
    if poly is not None:
        X_test = poly.transform(X_test)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return y_pred, mse, mae

def regression_pipeline(df, target='profit', test_size=0.3, random_state=42):
    # Select features (exclude target & non-numeric columns)
    X = df.drop(columns=[target, 'product_id', 'product_name', 'category'], errors='ignore')
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Linear Regression
    lin_model = train_linear_regression(X_train, y_train)
    lin_pred, lin_mse, lin_mae = evaluate_model(lin_model, X_test, y_test)

    # Polynomial Regression (degree 2)
    poly_model, poly_transformer = train_polynomial_regression(X_train, y_train, degree=2)
    poly_pred, poly_mse, poly_mae = evaluate_model(poly_model, X_test, y_test, poly=poly_transformer)

    results = {
        'linear': {'model': lin_model, 'pred': lin_pred, 'mse': lin_mse, 'mae': lin_mae},
        'polynomial': {'model': poly_model, 'pred': poly_pred, 'mse': poly_mse, 'mae': poly_mae},
        'X_test': X_test,
        'y_test': y_test
    }

    return results
