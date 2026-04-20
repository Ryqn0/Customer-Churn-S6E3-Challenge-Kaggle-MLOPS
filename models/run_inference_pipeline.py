"""
This script runs the inference pipeline for a given model and dataset. It takes in the model name, dataset name, and output directory as arguments, and saves the inference results to the specified output directory. The script performs the following steps:
1. Loads the specified model from the models directory.
2. Loads the specified dataset.
3. Runs inference on the dataset using the loaded model.
4. Saves the inference results to the specified output directory.
Usage:
    python run_inference_pipeline.py --model_name <model_name> --dataset_name <dataset_name> --output_dir <output_dir>
"""

import os
import argparse
import json
from pathlib import Path
import joblib
import mlflow
import pandas as pd

from data.preprocess import preprocess_data
from features.build_features import build_features
from utils.validate_data import validate_data

INFERENCE_COLUMN_RENAME_MAP = {
    "senior_citizen": "SeniorCitizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "phone_service": "PhoneService",
    "multiple_lines": "MultipleLines",
    "internet_service": "InternetService",
    "online_security": "OnlineSecurity",
    "online_backup": "OnlineBackup",
    "device_protection": "DeviceProtection",
    "tech_support": "TechSupport",
    "streaming_tv": "StreamingTV",
    "streaming_movies": "StreamingMovies",
    "contract": "Contract",
    "paperless_billing": "PaperlessBilling",
    "payment_method": "PaymentMethod",
    "monthly_charges": "MonthlyCharges",
    "total_charges": "TotalCharges",
}


def _load_training_feature_names(project_root: Path) -> list[str] | None:
    feature_names_path = project_root / "data" / "processed" / "feature_names.json"
    if not feature_names_path.exists():
        return None

    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    # Keep only model input features.
    return [name for name in feature_names if name != "Churn"]


def _align_features_to_training_schema(data: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    expected_features = _load_training_feature_names(project_root)
    if not expected_features:
        return data

    return data.reindex(columns=expected_features, fill_value=0)

# Function to load the model
def load_model(model_name: str):
    project_root = Path(__file__).resolve().parents[1]
    model_filename = f"{model_name}.pkl"

    # Search canonical project locations first, then cwd-relative fallbacks.
    candidate_paths = [
        project_root / "models" / model_filename,
        project_root / "src" / "models" / model_filename,
        Path.cwd() / "models" / model_filename,
        Path.cwd() / "src" / "models" / model_filename,
    ]

    for model_path in candidate_paths:
        if model_path.exists():
            model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            return model

    searched_paths = "\n".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        f"Model file '{model_filename}' not found. Searched:\n{searched_paths}"
    )


    return model

# Function to predict using the model
def predict(model, data: dict) -> str:

    if not hasattr(model, "predict"):
        raise ValueError("The loaded model does not have a predict method.")

    project_root = Path(__file__).resolve().parents[1]
    
    # Apply transformation to the data if necessary (e.g., feature engineering, scaling)
    print("Running inference on the input data...")
    data = pd.DataFrame(data, index=[0])  # Convert input data to DataFrame for processing
    data = data.rename(columns=INFERENCE_COLUMN_RENAME_MAP)
    print(f"Input data shape: {data.shape}")
    data = preprocess_data(data)
    print(f"Data shape after preprocessing: {data.shape}")

    is_valid, validation_errors = validate_data(data)
    if not is_valid:
        raise ValueError(f"Input data validation failed: {validation_errors}")

    print("Input data validation passed.")
    data = build_features(data)
    data = _align_features_to_training_schema(data, project_root)
    print(f"Data shape after feature engineering: {data.shape}")

    predictions = model.predict(data)
    print("Inference completed successfully.")

    prediction_value = predictions[0] if hasattr(predictions, "__getitem__") else predictions
    if int(prediction_value) == 1:
        return "The customer is likely to churn."
    else:
        return "The customer is not likely to churn."