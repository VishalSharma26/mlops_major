import os
import sys
import joblib
from utils import load_data

BASE_DIR   = os.path.join(os.path.dirname(__file__), "../")
MODEL_FILE = os.path.abspath(os.path.join(BASE_DIR, "model", "linear_model.joblib"))

if not os.path.exists(os.path.dirname(MODEL_FILE)):
    os.makedirs(os.path.dirname(MODEL_FILE))

def load():
    return joblib.load(MODEL_FILE)

def predict():
    # Load data
    _, X_test, _, y_test = load_data()

    # Load model
    model = load()

    # Predict
    predictions = model.predict(X_test[:5])
    print("Sample Predictions:", predictions)
    print("Actual Values:", y_test[:5])

if __name__ == "__main__":
    predict()
