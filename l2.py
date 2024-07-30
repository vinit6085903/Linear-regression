import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load the dataset
url = 'https://raw.githubusercontent.com/krishnaik06/simple-Linear-Regression/master/Salary_Data.csv'
df = pd.read_csv(url)

# Define features and target
X = df[['YearsExperience']]
y = df['Salary']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge Regression with GridSearchCV
ridge = Ridge()
param_grid = {'alpha': [1, 2, 3, 4, 5]}  # The correct syntax for the parameter grid

# Setup the grid search
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best model
best_ridge = grid_search.best_estimator_

# Get user input
try:
    a = float(input("Enter the number of years of experience: "))
    new_data = pd.DataFrame({'YearsExperience': [a]})

    # Scale the new data
    new_data_scaled = scaler.transform(new_data)

    # Make prediction
    predicted_salary = best_ridge.predict(new_data_scaled)
    print(f"Predicted Salary for {a} years of experience: {predicted_salary[0]:,.2f}")
except ValueError:
    print("Invalid input. Please enter a numeric value.")
