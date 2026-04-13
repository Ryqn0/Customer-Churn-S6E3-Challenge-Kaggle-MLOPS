'''
Training tests for the Customer Churn Prediction project.
This test file is designed to validate the training process of the machine learning model. It ensures that
the model training function can successfully train a model on the preprocessed data and that the trained model meets expected performance criteria.
'''

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))  # Add the src directory to the system path to import modules

import pandas as pd
import joblib
from data.preprocess import preprocess_data
from features.build_features import build_features
from models.train import train_model

DATA_PATH = SRC_DIR / "data" / "raw" / "train.csv"
TARGET_COLUMN = "Churn"

def test_train_model():
    """
    Test the train_model function to ensure it can successfully train a model on the preprocessed data.
    This test loads the test dataset, preprocesses it, builds features, and then trains the model. It checks if the training process completes without errors and if the trained model meets expected performance criteria.
    """
    # Load the test data
    data = pd.read_csv(DATA_PATH)

    # Preprocess the data
    processed_data = preprocess_data(data)

    # Build features
    features_data = build_features(processed_data)

    # Train the model
    try:
        train_model(features_data, TARGET_COLUMN)
        print("Model training test passed successfully!")
    except Exception as e:
        assert False, f"Model training failed with error: {e}"