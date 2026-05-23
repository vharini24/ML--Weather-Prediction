# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: V. HARINI
RegisterNumber:  212225040113
*/
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

# Load dataset
df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")

# Check dataset
print(df.shape)
print(df.columns)

# Fill missing values
df = df.fillna(df.mean(numeric_only=True))

# Convert time column
df['time'] = pd.to_datetime(df['time'])

# Extract date features
df['Year'] = df['time'].dt.year
df['Month'] = df['time'].dt.month
df['Day'] = df['time'].dt.day
df['Hour'] = df['time'].dt.hour

# Features
X = df[['hum', 'pressure', 'wind_speed',
        'Year', 'Month', 'Day', 'Hour']]

# Targets
y = df[['tem', 'pm2_5', 'tsr']]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Evaluation")
print("---------------------")
print("R2 Score :", r2)
print("MAE      :", mae)
print("RMSE     :", rmse)

# Cross validation
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='r2'
)

print("\nCross Validation Scores:")
print(cv_scores)

print("\nAverage CV Score:", cv_scores.mean())








```

## Output:
<img width="728" height="345" alt="image" src="https://github.com/user-attachments/assets/774c8b04-69f6-4e04-b60f-528033a0f7fa" />

## Result:
