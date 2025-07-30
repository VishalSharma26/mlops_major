import os
import sys
import joblib
import numpy as np

BASE_DIR     = os.path.join(os.path.dirname(__file__), "../")
MODEL_FILE   = os.path.abspath(os.path.join(BASE_DIR, "model", "linear_model.joblib"))
UNQUANT_FILE = os.path.abspath(os.path.join(BASE_DIR, "model", "unquant_params.joblib"))
QUANT_FILE   = os.path.abspath(os.path.join(BASE_DIR, "model", "quant_params.joblib"))

if not os.path.exists(os.path.dirname(MODEL_FILE)):
    os.makedirs(os.path.dirname(MODEL_FILE))
    
def quantize_model():
    # Load trained model
    model = joblib.load(MODEL_FILE)

    # Extract parameters
    coef = model.coef_
    intercept = model.intercept_

    # Save unquantized parameters
    joblib.dump({"coef": coef, "intercept": intercept}, UNQUANT_FILE)
    print("Unquantized parameters saved.")

    # Quantization (scale and convert to uint8)
    scale = np.max(np.abs(coef)) / 255
    quant_coef = np.round(coef / scale).astype(np.uint8)
    quant_intercept = np.round(intercept / scale).astype(np.uint8)

    joblib.dump(
        {"coef": quant_coef, "intercept": quant_intercept, "scale": scale},
        QUANT_FILE
    )
    print("Quantized parameters saved.")

    # Dequantize and run inference
    dequant_coef = quant_coef.astype(np.float32) * scale
    dequant_intercept = float(quant_intercept) * scale

    # Create a copy of the model with dequantized params
    from sklearn.linear_model import LinearRegression
    quant_model = LinearRegression()
    quant_model.coef_ = dequant_coef
    quant_model.intercept_ = dequant_intercept

    print("Quantization completed successfully.")
    return quant_model

if __name__ == "__main__":
    quantize_model()
