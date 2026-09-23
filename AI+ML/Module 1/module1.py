from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

csv_path = Path(__file__).resolve().parent / "50_Startups.csv"


file = pd.read_csv(csv_path)
print(file.head())
file.info()
print(file.describe())


# ---------------------------------------------------------------
# 1. Process the dataset
# ---------------------------------------------------------------
# "State" is text, so turn it into 0/1 columns (one-hot encoding).
# drop_first=True drops one state to avoid redundant columns.
data = pd.get_dummies(file, columns=["State"], drop_first=True, dtype=float)

X = data.drop(columns=["Profit"])
y = data["Profit"]
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42
)

# Standardize features (mean 0, std 1). Gradient descent needs this to
# converge, since R&D/Marketing are in the 100,000s but dummies are 0/1.
# Use the training set's mean/std for both sets so no test data leaks in.
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# ---------------------------------------------------------------
# 2. Linear Regression with scikit-learn
# ---------------------------------------------------------------
sk_model = LinearRegression()
sk_model.fit(X_train, y_train)
sk_pred = sk_model.predict(X_test)


# ---------------------------------------------------------------
# 3. Linear Regression from scratch (gradient descent)
# ---------------------------------------------------------------
def gradient_descent(X, y, learning_rate=0.01, epochs=10000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for epoch in range(epochs):
        y_pred = X @ weights + bias
        error = y_pred - y

        # Gradients of the mean squared error with respect to weights and bias
        dw = (2 / n_samples) * (X.T @ error)
        db = (2 / n_samples) * error.sum()

        weights -= learning_rate * dw
        bias -= learning_rate * db

        if epoch % 2000 == 0:
            print(f"Epoch {epoch:5d}  MSE = {np.mean(error ** 2):,.2f}")

    return weights, bias


def predict(X, weights, bias):
    return X @ weights + bias


print("\nTraining custom model...")
my_weights, my_bias = gradient_descent(X_train, y_train)
my_pred = predict(X_test, my_weights, my_bias)


# ---------------------------------------------------------------
# 4. Compare coefficients and intercept
# ---------------------------------------------------------------
comparison = pd.DataFrame(
    {"scikit-learn": sk_model.coef_, "From scratch": my_weights},
    index=feature_names,
)
comparison.loc["Intercept"] = [sk_model.intercept_, my_bias]
comparison["Difference"] = comparison["scikit-learn"] - comparison["From scratch"]

print("\nCoefficients (on standardized features):")
print(comparison.round(2))

print(f"\nR² on test set - scikit-learn:  {r2_score(y_test, sk_pred):.4f}")
print(f"R² on test set - From scratch:  {r2_score(y_test, my_pred):.4f}")
