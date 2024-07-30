import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load the dataset
url = 'https://raw.githubusercontent.com/nicolelumagui/ML-Exercise_Advertising_Linear-Regression/master/IPYNB%20and%20Dataset/Advertising.csv'
df = pd.read_csv(url)

# Display columns
print("Columns in the dataset:")
print(df.columns)

# Define features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

# Print the shape of the datasets
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Testing target shape: {y_test.shape}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge Regression with GridSearchCV
ridge = Ridge()
param_grid = {'alpha': [0.1, 1, 10, 100]}  # You can adjust these values based on your needs

# Setup the grid search
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best parameters and best score
best_params = grid_search.best_params_
best_score = -grid_search.best_score_  # Convert negative score to positive
print(f"Best Parameters: {best_params}")
print(f"Best CV Score (Mean Squared Error): {best_score}")

# Use the best model to make predictions
best_ridge = grid_search.best_estimator_
y_pred = best_ridge.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line for perfect prediction
plt.show()
def predict_new_data(new_data):
    new_data_df = pd.DataFrame(new_data, columns=['TV', 'Radio', 'Newspaper'])
    new_data_scaled = scaler.transform(new_data_df)
    predictions = best_ridge.predict(new_data_scaled)
    
    return predictions
new_data = [
    {'TV': 100, 'Radio': 20, 'Newspaper': 30},
    {'TV': 200, 'Radio': 40, 'Newspaper': 60}
]

predictions = predict_new_data(new_data)
for i, pred in enumerate(predictions):
    print(f"Predicted Sales for new data {i+1}: ${pred:.2f}")