import pytest
import pandas as pd
from src.data.preprocess import preprocess_data

def test_preprocess_data():
    """
    Test the preprocess_data function to ensure it correctly handles missing values and encodes categorical variables.
    This test creates a sample DataFrame with missing values and categorical variables, then checks if the preprocessing function correctly fills missing values and encodes the target variable.
    """
    # Create a sample DataFrame with missing values and categorical variables
    data = pd.DataFrame({
        "feature1": [1, 2, None, 4],
        "feature2": ["A", "B", "A", None],
        "target": [0, 1, 1, 0]
    })

    # Call the preprocess_data function
    processed_data = preprocess_data(data)

    # Check if missing values are filled
    assert not processed_data.isnull().any().any()

    # Check if categorical variables are encoded
    assert processed_data["feature2"].dtype == "int64"

    # Check if the target variable is encoded
    assert processed_data["target"].dtype == "int64"