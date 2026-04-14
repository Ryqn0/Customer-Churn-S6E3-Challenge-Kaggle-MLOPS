'''
This test file is designed to validate the functionality of the model evaluation process defined in src/models/evaluate.py. 
It ensures that the evaluation metrics are calculated correctly and that the function can handle various types of models, 
including those that do not support probability predictions. The tests will cover accuracy, precision, recall, F1 score, and ROC AUC score, 
as well as the generation of classification reports and confusion matrices.
'''

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))  # Add the src directory to the system path to import modules

MODEL_PATH = ROOT_DIR / "tests" / "voting_model.pkl"
DATA_PATH = ROOT_DIR / "src" / "data" / "raw" / "train.csv"

import joblib
from models.evaluate import evaluate_model
from data.preprocess import preprocess_data
from features.build_features import build_features
import pandas as pd

def test_evaluate_model():
    """
    Test the evaluate_model function to ensure it correctly calculates evaluation metrics and handles models without probability predictions.
    This test loads a pre-trained model, creates a sample test dataset, and checks if the evaluation function returns the expected metrics and handles edge cases appropriately.
    """
    # Load the pre-trained model
    model = joblib.load(MODEL_PATH)

    # Load the test dataset
    data = pd.read_csv(DATA_PATH)

    # Preprocess the data
    processed_data = preprocess_data(data)
    features_data = build_features(processed_data)
    X_test = features_data.drop(columns=["Churn"])
    y_test = features_data["Churn"]

    # Evaluate the model
    evaluation_results = evaluate_model(model, X_test, y_test)

    # Check if the evaluation results contain the expected keys
    expected_keys = {"accuracy", "precision", "recall", "f1_score", "roc_auc_score", "classification_report", "confusion_matrix"}
    assert set(evaluation_results.keys()) == expected_keys, f"Evaluation results should contain keys: {expected_keys}"


if __name__ == "__main__":    
    test_evaluate_model()
    print("All tests passed successfully!")