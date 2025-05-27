import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

def predict_next_six_months(csv_path, model, scaler_X, scaler_y, device='cpu'):
    # Load and prepare data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}. Please ensure the file exists.")
        return None

    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str), format='%Y-%m')
    df = df.set_index('Date').sort_index()

    # Debug: Print initial data
    print("\nInitial DataFrame:")
    print(df.tail())

    # Feature engineering: Create lagged features
    lags = [1, 2, 3, 6, 12]
    for lag in lags:
        df[f'Units_Sold_Lag_{lag}'] = df['Units_Sold'].shift(lag)

    # Feature engineering: Add time-based features
    df['Month_of_Year'] = df.index.month
    df['Year_Numerical'] = df.index.year

    # Debug: Print DataFrame with lagged features
    print("\nDataFrame with lagged features:")
    print(df.tail(15))

    # Handle missing values in historical data
    df = df.fillna(method='ffill').fillna(method='bfill')
    print("\nDataFrame after filling NaNs in historical data:")
    print(df.tail())

    # Initialize lists to store predictions
    predictions = []
    last_date = df.index.max()
    future_dates = [last_date + pd.offsets.MonthEnd(n) + timedelta(days=1) for n in range(1, 7)]

    # Features used in training
    features = [f'Units_Sold_Lag_{lag}' for lag in lags] + ['Month_of_Year', 'Year_Numerical']

    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()

    # Initialize current_df with all necessary columns
    current_df = df.copy()
    for col in features:
        if col not in current_df.columns:
            current_df[col] = np.nan  # Ensure all feature columns exist

    # Predict for the next 6 months
    for future_date in future_dates:
        print(f"\nGenerating features for {future_date.strftime('%Y-%m')}:")
        # Create a new row for the future date
        new_row = pd.DataFrame(index=[future_date], columns=current_df.columns)
        new_row['Month_of_Year'] = future_date.month
        new_row['Year_Numerical'] = future_date.year

        # Fill lagged features
        for lag in lags:
            lag_date = future_date - pd.offsets.MonthEnd(lag)
            if lag_date in current_df.index:
                new_row[f'Units_Sold_Lag_{lag}'] = current_df.loc[lag_date, 'Units_Sold']
            else:
                # Use the most recent predicted or historical value
                for i in range(1, lag + 1):
                    prev_date = future_date - pd.offsets.MonthEnd(i)
                    if prev_date in current_df.index:
                        new_row[f'Units_Sold_Lag_{lag}'] = current_df.loc[prev_date, 'Units_Sold']
                        break
                else:
                    # Fallback to the last historical value
                    new_row[f'Units_Sold_Lag_{lag}'] = current_df['Units_Sold'].iloc[-1]

        # Debug: Print features for the current prediction
        print("Features for prediction:")
        print(new_row[features])

        # Check for NaN in features
        if new_row[features].isna().any().any():
            print(f"Warning: NaN values found in features for {future_date.strftime('%Y-%m')}")
            print(new_row[features])
            new_row[features] = new_row[features].fillna(current_df[features].iloc[-1])

        # Scale the features
        try:
            X_new = scaler_X.transform(new_row[features])
        except Exception as e:
            print(f"Error scaling features for {future_date.strftime('%Y-%m')}: {e}")
            return None

        X_new_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)

        # Make prediction
        with torch.inference_mode():
            pred_scaled = model(X_new_tensor).cpu().numpy()

        # Debug: Print scaled prediction
        print(f"Scaled prediction: {pred_scaled}")

        # Reverse scale the prediction
        try:
            pred = scaler_y.inverse_transform(pred_scaled)[0, 0]
        except Exception as e:
            print(f"Error inverse scaling prediction for {future_date.strftime('%Y-%m')}: {e}")
            return None

        # Debug: Print unscaled prediction
        print(f"Unscaled prediction: {pred}")

        # Append prediction to results
        predictions.append({'Date': future_date, 'Predicted_Units_Sold': pred})

        # Update current_df with the new row
        new_row['Units_Sold'] = pred
        current_df = pd.concat([current_df, new_row], axis=0)

    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions)
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
    predictions_df.set_index('Date', inplace=True)

    # Print predictions
    print("\nPredicted Units Sold for the Next 6 Months:")
    for date, pred in predictions_df.iterrows():
        print(f"{date.strftime('%Y-%m')}: {pred['Predicted_Units_Sold']:.2f}")

    return predictions_df