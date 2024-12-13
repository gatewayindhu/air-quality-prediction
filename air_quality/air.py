import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Dataset Preparation
# Simulate a dataset for AQI prediction (use a real dataset for production)
data = {
    'PM2.5': np.random.uniform(10, 250, 500),
    'PM10': np.random.uniform(20, 300, 500),
    'NO2': np.random.uniform(5, 100, 500),
    'SO2': np.random.uniform(2, 50, 500),
    'AQI': np.random.uniform(50, 400, 500),
}
data = pd.DataFrame(data)

# Step 2: Split data into training and testing sets
X = data[['PM2.5', 'PM10', 'NO2', 'SO2']]
y = data['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

def predict_aqi(pm25, pm10, no2, so2):
    input_data = np.array([[pm25, pm10, no2, so2]])
    prediction = model.predict(input_data)[0]
    return round(prediction, 2)

# Step 4: Console-based AQI Predictor
def main():
    print("Welcome to the AQI Predictor")
    try:
        pm25 = float(input("Enter PM2.5 (μg/m3): "))
        pm10 = float(input("Enter PM10 (μg/m3): "))
        no2 = float(input("Enter NO2 (ppb): "))
        so2 = float(input("Enter SO2 (ppb): "))

        # Predict AQI
        prediction = predict_aqi(pm25, pm10, no2, so2)
        print(f"Predicted AQI: {prediction}")

        # Visualize the relationship between PM2.5 and AQI
        plt.figure(figsize=(10, 6))
        plt.scatter(data['PM2.5'], data['AQI'], alpha=0.6, color='blue', label='Data')
        plt.xlabel('PM2.5 (μg/m³)')
        plt.ylabel('AQI')
        plt.title('PM2.5 vs AQI')
        plt.legend()
        plt.show()

        # Plot residuals (errors)
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6, color='red')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.xlabel('Predicted AQI')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.title('Residual Plot')
        plt.show()

    except ValueError:
        print("Invalid input. Please enter numerical values.")

if __name__ == "__main__":
    main()
