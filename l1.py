import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
url = 'https://raw.githubusercontent.com/krishnaik06/simple-Linear-Regression/master/Salary_Data.csv'
df = pd.read_csv(url)
print(df.head())
print(df.isnull().sum())
X = df[['YearsExperience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
regression = LinearRegression()
regression.fit(X_train_scaled, y_train)
cv_scores = cross_val_score(regression, X_train_scaled, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
ridge = Ridge()
param_grid = {'alpha': [1, 2, 3, 4, 5]}  # The correct syntax for the parameter grid
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best CV Score: {-best_score}")
best_ridge = grid_search.best_estimator_
y_pred = best_ridge.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Ridge Regression Predictions')
plt.legend()
plt.show()
