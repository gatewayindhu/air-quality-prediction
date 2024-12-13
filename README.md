# air-quality-prediction
This code implements an AQI prediction tool based on simulated environmental pollutant data. It leverages a Linear Regression model to predict AQI using pollutant levels as input

Step-by-Step Description
Libraries and Dataset Preparation:

Required libraries (pandas, numpy, matplotlib, and scikit-learn) are imported.
A synthetic dataset is generated with pollutant levels (PM2.5, PM10, NO2, and SO2) as independent variables and AQI as the target variable.
Data Splitting:

Features (PM2.5, PM10, NO2, SO2) are stored in X, and the target variable (AQI) in y.
The dataset is split into training (80%) and testing (20%) sets using train_test_split.
Model Training:

A Linear Regression model from sklearn is trained on the training data (X_train, y_train) to learn the relationship between pollutants and AQI.
AQI Prediction Function:

A function, predict_aqi, takes pollutant levels as input and predicts the AQI using the trained regression model.
Console-Based AQI Predictor:

User Input: Prompts the user to input pollutant levels.
AQI Prediction: Calls predict_aqi to calculate and display the predicted AQI.
Data Visualization:
PM2.5 vs AQI: A scatter plot shows the relationship between PM2.5 levels and AQI.
Residual Plot: Visualizes the prediction errors (Residuals = Actual - Predicted) to evaluate model performance.
Error Handling:

Ensures user inputs are valid numerical values. If not, it displays an error message.
