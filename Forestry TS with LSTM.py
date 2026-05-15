"""Generated from Jupyter notebook: Forestry TS with LSTM

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA


def main():
    # --- Data Collection ---

    # Fetch NASA FIRMS Fire Data
    fire_data_url = "https://firms.modaps.eosdis.nasa.gov/api/download/"
    fire_df = pd.read_csv(fire_data_url)  # Replace with actual API or file location

    # Fetch NOAA Climate Data (Alternative: Use `climateR` or download manually)
    climate_data_url = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"
    response = requests.get(climate_data_url)
    if response.status_code == 200:
        climate_df = pd.read_csv(StringIO(response.text))
    else:
        print("Failed to fetch NOAA data. Please download manually.")
        climate_df = pd.DataFrame()

    # Fetch Biomass Data from USFS FIA (Manually Downloaded)
    biomass_file = "FIA_Biomass_Data.csv"  # Ensure this file is downloaded
    try:
        biomass_df = pd.read_csv(biomass_file)
    except FileNotFoundError:
        print("Biomass data file not found. Please download manually.")
        biomass_df = pd.DataFrame()

    # --- Data Preprocessing ---
    for df in [fire_df, climate_df, biomass_df]:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    # Merge available datasets
    df = (
        pd.merge(fire_df, climate_df, on="date", how="inner")
        if not climate_df.empty
        else fire_df
    )
    df = pd.merge(df, biomass_df, on="date", how="inner") if not biomass_df.empty else df

    # Handle missing values
    df.fillna(method="ffill", inplace=True)

    # --- Data Visualization ---
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["fire_occurrences"], label="Wildfires")
    plt.plot(df["date"], df["temperature"], label="Temperature")
    plt.xlabel("Year")
    plt.ylabel("Occurrences / Temperature")
    plt.title("Wildfires vs. Temperature Over Time")
    plt.legend()
    plt.savefig("wildfire_temperature_trend.png")
    plt.show()

    # --- ARIMA Model for Biomass Forecasting ---
    if "biomass" in df.columns:
        arima_model = ARIMA(df["biomass"], order=(5, 1, 0))
        arima_result = arima_model.fit()
        forecast = arima_result.forecast(steps=12)

        plt.figure(figsize=(10, 5))
        plt.plot(df["date"], df["biomass"], label="Observed Biomass")
        plt.plot(
            pd.date_range(df["date"].iloc[-1], periods=12, freq="M"),
            forecast,
            label="Forecasted Biomass",
            linestyle="dashed",
        )
        plt.xlabel("Year")
        plt.ylabel("Biomass")
        plt.legend()
        plt.savefig("biomass_forecast.png")
        plt.show()

    # --- Machine Learning Model for Fire Risk Prediction ---
    features = ["temperature", "humidity", "wind_speed"]
    if all(feature in df.columns for feature in features):
        X = df[features]
        y = df["fire_occurrences"] > 0  # Binary classification

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)
        print(f"Fire Risk Prediction Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # --- Deep Learning Model for NDVI Forecasting ---
    if "vegetation_index" in df.columns:

        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out

        # Prepare NDVI time series
        ndvi_data = df["vegetation_index"].values.reshape(-1, 1)
        X_ndvi = []
        y_ndvi = []
        seq_length = 12

        for i in range(len(ndvi_data) - seq_length):
            X_ndvi.append(ndvi_data[i : i + seq_length])
            y_ndvi.append(ndvi_data[i + seq_length])

        X_ndvi, y_ndvi = np.array(X_ndvi), np.array(y_ndvi)
        X_ndvi_train, X_ndvi_test = X_ndvi[:-12], X_ndvi[-12:]
        y_ndvi_train, y_ndvi_test = y_ndvi[:-12], y_ndvi[-12:]

        X_ndvi_train = torch.tensor(X_ndvi_train, dtype=torch.float32)
        y_ndvi_train = torch.tensor(y_ndvi_train, dtype=torch.float32)
        X_ndvi_test = torch.tensor(X_ndvi_test, dtype=torch.float32)
        y_ndvi_test = torch.tensor(y_ndvi_test, dtype=torch.float32)

        # Train LSTM Model
        model = LSTMModel(input_size=1, hidden_size=50, output_size=1, num_layers=2)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_ndvi_train)
            loss = criterion(output, y_ndvi_train)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

        # Predict NDVI
        model.eval()
        predicted_ndvi = model(X_ndvi_test).detach().numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(y_ndvi_test.numpy(), label="Actual NDVI")
        plt.plot(predicted_ndvi, label="Predicted NDVI", linestyle="dashed")
        plt.xlabel("Time")
        plt.ylabel("NDVI")
        plt.legend()
        plt.savefig("ndvi_forecast.png")
        plt.show()

    # --- Summary ---
    print("Forestry time series analysis complete. Results saved as images.")


if __name__ == "__main__":
    main()
