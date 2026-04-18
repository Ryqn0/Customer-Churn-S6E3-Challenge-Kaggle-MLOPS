import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
import joblib

def train_model(data : pd.DataFrame, target_column : str):
    """
    Train a machine learning model using XGBoost, LightGBM, and CatBoost classifiers, and evaluate their performance.

    Parameters:
    data (pd.DataFrame): The input dataset containing features and target variable.
    target_column (str): The name of the target variable column in the dataset.
    """

    print("Starting model training...")

    print(f"Target column: {target_column}")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    print("Training and testing data split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Initialize the classifiers
    print("Initializing classifiers...")
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]  # Calculate scale_pos_weight for XGBoost
    voting_model = VotingClassifier(
        estimators=[
            ("xgb", XGBClassifier(random_state=42, max_depth=5, n_estimators=500, learning_rate=0.1, scale_pos_weight=scale_pos_weight)),
            ("lgbm", LGBMClassifier(random_state=42, max_depth=6, n_estimators=500, learning_rate=0.1, scale_pos_weight=scale_pos_weight)),
            ("catboost", CatBoostClassifier(random_state=42, max_depth=6, n_estimators=600, learning_rate=0.1, scale_pos_weight=scale_pos_weight, verbose=0))
        ],
        voting="soft"
    )

    print("Classifiers initialized. Starting training...")

    #mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Training Experiment")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Voting Classifier"):

        # Log model parameters
        print("Logging model parameters to MLflow...")
        mlflow.log_params(voting_model.get_params())  # Log all parameters of the voting model

        # train the voting model
        print("Training the voting model...")
        voting_model.fit(X_train, y_train)

        # Make predictions and evaluate the model
        print("Making predictions and evaluating the model...")
        y_pred = voting_model.predict(X_test)
        y_proba = voting_model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        print("Calculating evaluation metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log evaluation metrics to MLflow
        print("Logging evaluation metrics to MLflow...")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Print evaluation metrics
        print("Model evaluation metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        # Save the trained model
        print("Saving the trained model...")
        joblib.dump(voting_model, "voting_model.pkl")
        mlflow.log_artifact("voting_model.pkl")
        print("Model training and evaluation completed successfully!")