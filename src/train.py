import os
import sys
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(__file__))
from utils import load_data

LINER_MODEL_FILE = os.path.abspath(os.path.join(BASE_DIR, "model", "linear_model.joblib"))

if not os.path.exists(os.path.dirname(LINER_MODEL_FILE)):
    os.makedirs(os.path.dirname(LINER_MODEL_FILE))

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
    joblib.dump(model, LINER_MODEL_FILE)
    print(f"Model saved as {LINER_MODEL_FILE}")

if __name__ == "__main__":
    train_model()
