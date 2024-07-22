import numpy as np
from sklearn.model_selection import train_test_split

from linear_regression import *

# Generating some data for demonstration
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4]])
y = np.dot(X, np.array([1, 2])) + 3

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the model
model = LinearRegression(learning_rate=0.01, n_iterations=1000, regularization='l2', alpha=0.1)
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

# Evaluating the model
mse = model.mean_squared_error(y_true=y_test, y_pred=predictions)
r2 = model.r2_score(y_true=y_test, y_pred=predictions)

print("Test Predictions:", predictions)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
