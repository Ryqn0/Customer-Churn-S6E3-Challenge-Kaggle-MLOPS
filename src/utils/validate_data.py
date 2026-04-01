import great_expectations as ge^
import pandas as pd
from typing import Tuple, List

def validate_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the input data using Great Expectations.

    Parameters:
    data (pd.DataFrame): The dataset to validate.

    Returns:
    Tuple[bool, List[str]]: A tuple containing a boolean indicating if the data is valid and a list of validation errors.
    """

    print("Validating data...")
    
    # Create a Great Expectations DataFrame
    ge_data = ge.from_pandas(data)

    # Define expectations for the dataset 

    # Expectation: 'id' column should exist and should not have null values
    ge_data.expect_column_to_exist('id')
    ge_data.expect_column_values_to_not_be_null('id')

    # Expectation: 'gender' column should exist and have two unique values
    ge_data.expect_column_to_exist('gender')
    ge_data.expect_column_values_to_be_in_set('gender', ['Male', 'Female'])

    # Expectation: 'SeniorCitizen' column should exist and have values 0 or 1
    ge_data.expect_column_to_exist('SeniorCitizen')
    ge_data.expect_column_values_to_be_in_set('SeniorCitizen', [0, 1])

    # Expectation: 'Partner' column should exist and have values 'Yes' or 'No'
    ge_data.expect_column_to_exist('Partner')
    ge_data.expect_column_values_to_be_in_set('Partner', ['Yes', 'No'])

    # Expectation: 'Dependents' column should exist and have values 'Yes' or 'No'
    ge_data.expect_column_to_exist('Dependents')
    ge_data.expect_column_values_to_be_in_set('Dependents', ['Yes', 'No'])

    # Expectation: 'tenure' column should exist and have non-negative integer values
    ge_data.expect_column_to_exist('tenure')
    ge_data.expect_column_values_to_be_of_type('tenure', 'int64')
    ge_data.expect_column_values_to_be_between('tenure', min_value=0, max_value=72)

    # Expectation: 'PhoneService' column should exist and have values 'Yes' or 'No'
    ge_data.expect_column_to_exist('PhoneService')
    ge_data.expect_column_values_to_be_in_set('PhoneService', ['Yes', 'No'])

    # Expectation: 'MultipleLines' column should exist and have values 'Yes', 'No', or 'No phone service'
    ge_data.expect_column_to_exist('MultipleLines')
    ge_data.expect_column_values_to_be_in_set('MultipleLines', ['Yes', 'No', 'No phone service'])

    # Expectation: 'InternetService' column should exist and have values 'DSL', 'Fiber optic', or 'No'
    ge_data.expect_column_to_exist('InternetService')
    ge_data.expect_column_values_to_be_in_set('InternetService', ['DSL', 'Fiber optic', 'No'])

    # Expectation: 'OnlineSecurity' column should exist and have values 'Yes', 'No', or 'No internet service'
    ge_data.expect_column_to_exist('OnlineSecurity')
    ge_data.expect_column_values_to_be_in_set('OnlineSecurity', ['Yes', 'No', 'No internet service'])

    # Expectation: 'OnlineBackup' column should exist and have values 'Yes', 'No', or 'No internet service'
    ge_data.expect_column_to_exist('OnlineBackup')
    ge_data.expect_column_values_to_be_in_set('OnlineBackup', ['Yes', 'No', 'No internet service'])

    # Expectation: 'DeviceProtection' column should exist and have values 'Yes', 'No', or 'No internet service'
    ge_data.expect_column_to_exist('DeviceProtection')
    ge_data.expect_column_values_to_be_in_set('DeviceProtection', ['Yes', 'No', 'No internet service'])

    # Expectation: 'TechSupport' column should exist and have values 'Yes', 'No', or 'No internet service'
    ge_data.expect_column_to_exist('TechSupport')
    ge_data.expect_column_values_to_be_in_set('TechSupport', ['Yes', 'No', 'No internet service'])

    # Expectation: 'StreamingTV' column should exist and have values 'Yes', 'No', or 'No internet service'
    ge_data.expect_column_to_exist('StreamingTV')
    ge_data.expect_column_values_to_be_in_set('StreamingTV', ['Yes', 'No', 'No internet service'])

    # Expectation: 'StreamingMovies' column should exist and have values 'Yes', 'No', or 'No internet service'
    ge_data.expect_column_to_exist('StreamingMovies')
    ge_data.expect_column_values_to_be_in_set('StreamingMovies', ['Yes', 'No', 'No internet service'])

    # Expectation: 'Contract' column should exist and have values 'Month-to-month', 'One year', or 'Two year'
    ge_data.expect_column_to_exist('Contract')
    ge_data.expect_column_values_to_be_in_set('Contract', ['Month-to-month', 'One year', 'Two year'])

    # Expectation: 'PaperlessBilling' column should exist and have values 'Yes' or 'No'
    ge_data.expect_column_to_exist('PaperlessBilling')
    ge_data.expect_column_values_to_be_in_set('PaperlessBilling', ['Yes', 'No'])

    # Expectation: 'PaymentMethod' column should exist and have values 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', or 'Credit card (automatic)'
    ge_data.expect_column_to_exist('PaymentMethod')
    ge_data.expect_column_values_to_be_in_set('PaymentMethod', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

    # Expectation: 'MonthlyCharges' column should exist and have non-negative float values
    ge_data.expect_column_to_exist('MonthlyCharges')
    ge_data.expect_column_values_to_be_of_type('MonthlyCharges', 'float64')
    ge_data.expect_column_values_to_be_between('MonthlyCharges', min_value=0, max_value=250)

    # Expectation: 'TotalCharges' column should exist and have non-negative float values
    ge_data.expect_column_to_exist('TotalCharges')
    ge_data.expect_column_values_to_be_of_type('TotalCharges', 'float64')
    ge_data.expect_column_values_to_be_between('TotalCharges', min_value=0, max_value=15000)

    # Validate the data and collect validation errors
    ge_data.expect_column_pair_values_A_to_be_greater_than_B('TotalCharges', 'MonthlyCharges', or_equal=True, mostly=0.95) # TotalCharges should be greater than or equal to MonthlyCharges

    print("Data validation completed.")

    validation_results = ge_data.validate()

    # Extract validation results
    is_valid = validation_results['success'] # Overall success of the validation
    print(f"Validation success: {is_valid}")

    # Extract validation errors
    validation_errors = [result['expectation_config']['expectation_type'] for result in validation_results['results'] if not result['success']]
    print(f"Validation errors: {validation_errors}")

    # Summarize validation results
    print("Validation summary:")
    for result in validation_results['results']:
        expectation_type = result['expectation_config']['expectation_type']
        success = result['success']
        print(f"{expectation_type}: {'Passed' if success else 'Failed'}")

    return is_valid, validation_errors
