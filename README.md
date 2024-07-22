# Linear Regression from Scratch in Python

## Overview

This repository contains a custom implementation of a linear regression model from scratch using Python. The implementation includes feature normalization, L2 regularization, gradient descent optimization, and provides a full end-to-end solution with data splitting, model training, and evaluation.

## Features

- **Feature Normalization**: Standardizes the features for better performance.
- **L2 Regularization**: Prevents overfitting by penalizing large coefficients.
- **Gradient Descent Optimization**: Iteratively adjusts model parameters to minimize the loss.
- **Evaluation Metrics**: Includes Mean Squared Error (MSE) and R2 Score for model evaluation.

## Implementation Details

### Linear Regression Class

The `LinearRegression` class includes methods for fitting the model, making predictions, and evaluating the model performance. Key methods include:

- `fit(X: np.ndarray, y: np.ndarray) -> None`: Trains the model using the training data.
- `predict(X: np.ndarray) -> np.ndarray`: Makes predictions on new data.
- `mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float`: Calculates the mean squared error.
- `r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float`: Calculates the R2 score.

## Usage

1. **Generate Sample Data**:
    ```python
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4]])
    y = np.dot(X, np.array([1, 2])) + 3
    ```

2. **Train the Model**:
    ```python
    model = LinearRegression(learning_rate=0.01, n_iterations=1000, regularization='l2', alpha=0.1)
    model.fit(X_train, y_train)
    ```

3. **Make Predictions**:
    ```python
    predictions = model.predict(X_test)
    ```

4. **Evaluate the Model**:
    ```python
    mse = model.mean_squared_error(y_test, predictions)
    r2 = model.r2_score(y_test, predictions)
    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)
    ```

## Example

An example usage script is provided in the `test.py` file. Run the script to see the model in action.

## Acknowledgements

This implementation is inspired by the concepts and techniques used in popular machine learning libraries like scikit-learn.
