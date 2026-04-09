import pandas as pd
from src.data.preprocess import preprocess_data
from src.utils.validate_data import validate_data

DATA_PATH = "../src/data/test.csv"
TARGET_COLUMN = "Churn"
FEATURES_LIST = ["id", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]

def test_preprocess_data():
    """
    Test the preprocess_data function to ensure it correctly handles missing values and encodes categorical variables.
    This test creates a sample DataFrame with missing values and categorical variables, then checks if the preprocessing function correctly fills missing values and encodes the target variable.
    """
    # Load the test data
    data = pd.read_csv(DATA_PATH)

    # Call the preprocess_data function
    processed_data = preprocess_data(data)
    assert processed_data is not None, "Preprocessed data should not be None"

    # Use the validate_data function to check if the processed data meets the expected schema and value constraints
    is_valid, validation_errors = validate_data(processed_data)
    assert is_valid == True, f"Data validation failed with errors: {validation_errors}"

    if TARGET_COLUMN in processed_data.columns:
    # Check if the target variable is encoded
        assert processed_data[TARGET_COLUMN].dtype == "int64" # Assuming the target variable is encoded as integers
        assert set(processed_data[TARGET_COLUMN].unique()) == {0, 1} # Assuming binary classification with 0 and 1 as encoded values


if __name__ == "__main__":    
    test_preprocess_data()
    print("All tests passed successfully!")