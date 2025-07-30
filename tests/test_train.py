import os
import sys
import joblib
import pytest
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(BASE_DIR)
from src.utils import load_data
from src.train import train_model

MODEL_FILE                = os.path.abspath(os.path.join(BASE_DIR, "model", "linear_model.joblib"))
MIN_PERFORMANCE_THRESHOLD = 0.5

def test_data_loading():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0

def test_model_training_and_saving():
    train_model()
    model = joblib.load(MODEL_FILE)
    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_")

def test_model_r2_score():
    X_train, X_test, y_train, y_test = load_data()
    model = joblib.load(MODEL_FILE)
    r2 = model.score(X_test, y_test)
    assert r2 > MIN_PERFORMANCE_THRESHOLD
