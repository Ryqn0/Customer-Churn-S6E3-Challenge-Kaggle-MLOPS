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
import joblib
import mlflow
import pandas as pd

from data.preprocess import preprocess_data
from features.build_features import build_features
from utils.validate_data import validate_data

# Function to load the model
def load_model(model_name: str):
    
    model_path = os.path.join("models", f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
    return model

# Function to predict using the model
def predict(model, data: dict) -> str:

    if not hasattr(model, "predict"):
        raise ValueError("The loaded model does not have a predict method.")
    
    # Apply transformation to the data if necessary (e.g., feature engineering, scaling)
    print("Running inference on the input data...")
    data = pd.DataFrame(data, index=[0])  # Convert input data to DataFrame for processing
    print(f"Input data shape: {data.shape}")
    data = preprocess_data(data)
    print(f"Data shape after preprocessing: {data.shape}")
    data = build_features(data)
    print(f"Data shape after feature engineering: {data.shape}")
    data = validate_data(data)
    print(f"Data shape after validation: {data.shape}")

    predictions = model.predict(data)
    print("Inference completed successfully.")

    if predictions == 1 :
        return "The customer is likely to churn."
    else:
        return "The customer is not likely to churn."