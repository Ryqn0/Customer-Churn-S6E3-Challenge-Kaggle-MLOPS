"""
File to run the entire pipeline for training and evaluating the model.
Process: 
1. Load the data
2. Validate the data
3. Preprocess the data
4. Feature engineering
5. Train the model
6. Evaluate the model
"""

import os
import sys
import time
import argparse
import numbers
from pathlib import Path
import pandas as pd
import mlflow
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

# Allow importing from src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.load_data import load_data
from utils.validate_data import validate_data
from data.preprocess import preprocess_data
from features.build_features import build_features
# from models.train import train_model
from models.evaluate import evaluate_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def normalize_tracking_uri(tracking_uri):
    if tracking_uri:
        if tracking_uri.startswith("file://") or "://" in tracking_uri:
            return tracking_uri
        return Path(tracking_uri).resolve().as_uri()
    return Path(os.path.join(PROJECT_ROOT, "mlruns")).resolve().as_uri()


def resolve_data_path(args):
    candidate_path = args.input if args.input else args.data_path
    if os.path.isabs(candidate_path):
        return candidate_path

    project_relative_path = os.path.join(PROJECT_ROOT, candidate_path)
    if os.path.exists(project_relative_path):
        return project_relative_path

    src_data_path = os.path.join(PROJECT_ROOT, 'src', candidate_path)
    if os.path.exists(src_data_path):
        return src_data_path

    src_raw_path = os.path.join(PROJECT_ROOT, 'src', 'data', 'raw', os.path.basename(candidate_path))
    if os.path.exists(src_raw_path):
        return src_raw_path

    return candidate_path


def log_numeric_metrics(metrics):
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, numbers.Real):
            mlflow.log_metric(metric_name, float(metric_value))

def main(args):
    """
    Main function to run the training pipeline.
    """

    # Configuring MLFLOW tracking URI and experiment name
    tracking_uri = normalize_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"Training Run - {time.strftime('%Y-%m-%d %H:%M:%S')}"):
        
        # Log parameters
        mlflow.log_param("model_type", "VotingClassifier with XGBoost, LightGBM, CatBoost")
        mlflow.log_param("target_column", args.target_column)
        mlflow.log_param("input_path", resolve_data_path(args))
        mlflow.log_param("test_size", args.test_size)

        # Data Loading and Validation
        print("Loading data...")
        data = load_data(resolve_data_path(args))
        print("Validating data...")
        is_valid, validation_errors = validate_data(data)
        print(f"Data validation result: {is_valid}")
        if not is_valid:
            print("Data validation failed with the following errors:")
            for error in validation_errors:
                print(f"- {error}")
            mlflow.log_param("data_validation", "Failed")
            mlflow.log_param("validation_errors", json.dumps(validation_errors))
            return
        mlflow.log_param("data_validation", "Passed")
        print(f"Data Shape: {data.shape}")

        # Data Preprocessing
        print("Preprocessing data...")
        data = preprocess_data(data, target_column=args.target_column)

        # Save the preprocessed data for future reference
        preprocessed_data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'preprocessed_data.csv')
        os.makedirs(os.path.dirname(preprocessed_data_path), exist_ok=True)
        data.to_csv(preprocessed_data_path, index=False)
        print(f"Preprocessed data saved to {preprocessed_data_path}")

        # Feature Engineering
        print("Building features...")
        data = build_features(data, target_column=args.target_column)
        print(f"Data Shape after feature engineering: {data.shape}")

        # Save the feature engineered data for future reference
        feature_engineered_data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'feature_engineered_data.csv')
        os.makedirs(os.path.dirname(feature_engineered_data_path), exist_ok=True)
        data.to_csv(feature_engineered_data_path, index=False)
        print(f"Feature engineered data saved to {feature_engineered_data_path}")

        # Save feature names metadata
        feature_names = data.columns.tolist()
        feature_names_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'feature_names.json')
        os.makedirs(os.path.dirname(feature_names_path), exist_ok=True)
        with open(feature_names_path, 'w') as f:
            json.dump(feature_names, f, indent=4)
        print(f"Feature names saved to {feature_names_path}")
        mlflow.log_artifact(feature_names_path, artifact_path="feature_names")

        # Save preprocessing artifacts
        preprocessing_artifacts_path = os.path.join(PROJECT_ROOT, 'artifacts', 'preprocessing')
        os.makedirs(preprocessing_artifacts_path, exist_ok=True)
        preprocessing_artifacts = {
            "preprocessed_data": preprocessed_data_path,
            "feature_columns": feature_names,
            "target_column": args.target_column
        }
        preprocessing_artifacts_file = os.path.join(preprocessing_artifacts_path, 'preprocessing_artifacts.json')
        with open(preprocessing_artifacts_file, 'w') as f:
            json.dump(preprocessing_artifacts, f, indent=4)
        print(f"Preprocessing artifacts saved to {preprocessing_artifacts_file}")
        mlflow.log_artifact(preprocessing_artifacts_file, artifact_path="preprocessing_artifacts")

        # Train-Test Split
        print("Splitting data into train and test sets...")
        X = data.drop(columns=[args.target_column])
        y = data[args.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)
        print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

        # Model Training
        print("Training model...")
        t0 = time.time()
        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]  # Calculate scale_pos_weight for XGBoost
        voting_model = VotingClassifier(
            estimators=[
                ("xgb", XGBClassifier(random_state=42, max_depth=5, n_estimators=500, learning_rate=0.1, scale_pos_weight=scale_pos_weight)),
                ("lgbm", LGBMClassifier(random_state=42, max_depth=6, n_estimators=500, learning_rate=0.1, scale_pos_weight=scale_pos_weight)),
                ("catboost", CatBoostClassifier(random_state=42, max_depth=6, n_estimators=600, learning_rate=0.1, scale_pos_weight=scale_pos_weight, verbose=0))
            ],
            voting="soft"
        )
        voting_model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("training_time_seconds", train_time)
        print(f"Model training completed in {train_time:.2f} seconds.")

        # Model Evaluation
        print("Evaluating model...")
        t1 = time.time()
        evaluation_results = evaluate_model(voting_model, X_test, y_test)
        evaluation_time = time.time() - t1
        mlflow.log_metric("evaluation_time_seconds", evaluation_time)
        print(f"Model evaluation completed in {evaluation_time:.2f} seconds.")
        log_numeric_metrics(evaluation_results)
        for metric_name, metric_value in evaluation_results.items():
            print(f"{metric_name}: {metric_value}")

        evaluation_report_path = os.path.join(PROJECT_ROOT, 'artifacts', 'evaluation_report.json')
        os.makedirs(os.path.dirname(evaluation_report_path), exist_ok=True)
        evaluation_report = {
            key: (value.tolist() if hasattr(value, 'tolist') else value)
            for key, value in evaluation_results.items()
        }
        with open(evaluation_report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=4)
        mlflow.log_artifact(evaluation_report_path, artifact_path="evaluation")

        # Model Saving
        model_path = os.path.join(PROJECT_ROOT, 'models', 'voting_model.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(voting_model, model_path)

        # MLFLOW model logging and serving
        mlflow.sklearn.log_model(voting_model, artifact_path="model", registered_model_name="VotingClassifierModel")
        print(f"Model saved to {model_path} and logged to MLflow.")
        print("Training pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training pipeline for customer churn prediction.")
    parser.add_argument("--input", type=str, default=None, help="Path to the input data file.")
    parser.add_argument("--data_path", type=str, default=os.path.join(PROJECT_ROOT, 'data', 'raw', 'train.csv'), help="Path to the raw data file.")
    parser.add_argument("--target", "--target_column", dest="target_column", type=str, default="Churn", help="Name of the target column in the dataset.")
    parser.add_argument("--experiment", type=str, default="Customer Churn Prediction")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None, help="Override the MLflow tracking URI. Use a local path for file tracking or http://127.0.0.1:5000 for a local MLflow server.")
    
    args = parser.parse_args()
    main(args)
