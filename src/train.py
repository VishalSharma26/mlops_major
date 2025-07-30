import os
import sys
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_data

def train_model():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Initialize and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"R² Score: {r2:.4f}")
    print(f"MSE Loss: {mse:.4f}")

    # Save model
    joblib.dump(model, "linear_model.joblib")
    print("Model saved as linear_model.joblib")

if __name__ == "__main__":
    train_model()
