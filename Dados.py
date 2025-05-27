import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Good practice for neural networks

def split_data(csv_path, targetCollumn, test_size, random_state):
    print('oi2')
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: csv file not found in {csv_path}. Please ensure the file is in the correct directory.")
        exit()

    # --- Step 1: Load and Prepare Data ---
    # Combine 'Year' and 'Month' to create a datetime index
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str), format='%Y-%m')
    df = df.set_index('Date').sort_index()

    # Display initial data info
    print("DataFrame after creating Date index:")
    print(df.head())
    print(df.info())

    # --- Step 3: Feature Engineering - Create Lagged Features ---
    # Define the lags you want to create
    lags = [1, 2, 3, 6, 12] # Lags for 1, 2, 3, 6, and 12 months prior

    for lag in lags:
        df[f'Units_Sold_Lag_{lag}'] = df['Units_Sold'].shift(lag)

    print("\nDataFrame with lagged features:")
    print(df.head(15)) # Show more rows to see lags filling in

    # --- Step 4: Feature Engineering - Add Time-Based Features ---
    df['Month_of_Year'] = df.index.month
    df['Year_Numerical'] = df.index.year

    print("\nDataFrame with time-based features:")
    print(df.head())

    # --- Step 5: Handle Missing Values (from Lags) ---
    # Drop rows with NaN values (which are created by the shift operation at the beginning of the series)
    df_cleaned = df.dropna()
    print(f"\nOriginal DataFrame size: {len(df)}")
    print(f"Cleaned DataFrame size after dropping NaNs: {len(df_cleaned)}")
    print("Cleaned DataFrame head:")
    print(df_cleaned.head())

    # --- Step 6: Define Target and Features ---
    # We will predict 'Units_Sold' as the primary target for this example
    target_variable = 'Units_Sold'

    # Features (X) will be the lagged values and time-based features
    features = [col for col in df_cleaned.columns if 'Lag_' in col]
    features.extend(['Month_of_Year', 'Year_Numerical'])

    X = df_cleaned[features]
    y = df_cleaned[target_variable]

    print(f"\nFeatures (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print("Sample Features (X) head:")
    print(X.head())

    # --- Step 7: Time Series Train-Test Split ---
    # Determine the split point: the last 6 months for prediction
    prediction_horizon_months = 6
    # Get the last date in the cleaned data
    last_date = df_cleaned.index.max()
    # Calculate the start date of the prediction horizon
    prediction_start_date = last_date - pd.DateOffset(months=prediction_horizon_months - 1)

    # Separate the data into training and prediction sets
    X_train_df = X[X.index < prediction_start_date]
    y_train_df = y[y.index < prediction_start_date]

    # Let's use the last 12 months of the training data as a validation set
    validation_months = 12
    validation_start_date = X_train_df.index.max() - pd.DateOffset(months=validation_months - 1)

    X_val_df = X_train_df[X_train_df.index >= validation_start_date]
    y_val_df = y_train_df[y_train_df.index >= validation_start_date]

    X_train_df = X_train_df[X_train_df.index < validation_start_date]
    y_train_df = y_train_df[y_train_df.index < validation_start_date]


    print(f"\nTraining data range: {X_train_df.index.min()} to {X_train_df.index.max()}")
    print(f"Validation data range: {X_val_df.index.min()} to {X_val_df.index.max()}")
    print(f"X_train_df shape: {X_train_df.shape}, y_train_df shape: {y_train_df.shape}")
    print(f"X_val_df shape: {X_val_df.shape}, y_val_df shape: {y_val_df.shape}")


    # --- Data Scaling (Good practice for PyTorch models) ---
    # Scale features to improve training stability and performance
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_df)
    X_val_scaled = scaler_X.transform(X_val_df)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_df.values.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val_df.values.reshape(-1, 1))

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_test = torch.tensor(y_val_scaled, dtype=torch.float32)

    return X_train, y_train, X_test, y_test, scaler_X, scaler_y