from typing import Any, List, Tuple

import great_expectations as gx
import pandas as pd
from great_expectations.core.batch import Batch
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.validator.validator import Validator

# Helper functions to extract success status and expectation type from validation results
def _extract_success(result: Any) -> bool:
    # Extract the success status from the validation result, handling both dictionary and object formats
    if isinstance(result, dict): 
        return bool(result.get("success", False))
    return bool(getattr(result, "success", False))

# Extract expectation type from validation result, handling both dictionary and object formats
def _extract_expectation_type(result: Any) -> str:
    # Extract the expectation type from the validation result, handling both dictionary and object formats
    if isinstance(result, dict):
        expectation_config = result.get("expectation_config", {})
        return expectation_config.get("type", expectation_config.get("expectation_type", "unknown_expectation"))

    # For object format, try to access expectation_config attribute and then type or expectation_type
    expectation_config = getattr(result, "expectation_config", None)
    # If expectation_config is None, return "unknown_expectation"
    if expectation_config is None:
        return "unknown_expectation"
    # If expectation_config is a dictionary, try to get type or expectation_type
    if isinstance(expectation_config, dict):
        return expectation_config.get("type", expectation_config.get("expectation_type", "unknown_expectation"))
    return getattr(expectation_config, "type", getattr(expectation_config, "expectation_type", "unknown_expectation"))

def validate_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the input data using Great Expectations.

    Parameters:
    data (pd.DataFrame): The dataset to validate.

    Returns:
    Tuple[bool, List[str]]: A tuple containing a boolean indicating if the data is valid and a list of validation errors.
    """

    print("Validating data...")
    
    context = gx.get_context(mode="ephemeral")
    validator = Validator(
        execution_engine=PandasExecutionEngine(),
        batches=[Batch(data=data)],
        data_context=context,
    )

    # Define expectations for the dataset 

    # Expectation: 'id' column should exist and should not have null values
    validator.expect_column_to_exist('id')
    validator.expect_column_values_to_not_be_null('id')

    # Expectation: 'gender' column should exist and have two unique values
    validator.expect_column_to_exist('gender')
    validator.expect_column_values_to_be_in_set('gender', ['Male', 'Female'])

    # Expectation: 'SeniorCitizen' column should exist and have values 'Yes' or 'No' after preprocessing
    validator.expect_column_to_exist('SeniorCitizen')
    validator.expect_column_values_to_be_in_set('SeniorCitizen', ['Yes', 'No'])

    # Expectation: 'Partner' column should exist and have values 'Yes' or 'No'
    validator.expect_column_to_exist('Partner')
    validator.expect_column_values_to_be_in_set('Partner', ['Yes', 'No'])

    # Expectation: 'Dependents' column should exist and have values 'Yes' or 'No'
    validator.expect_column_to_exist('Dependents')
    validator.expect_column_values_to_be_in_set('Dependents', ['Yes', 'No'])

    # Expectation: 'tenure' column should exist and have non-negative integer values
    validator.expect_column_to_exist('tenure')
    validator.expect_column_values_to_be_of_type('tenure', 'int64')
    validator.expect_column_values_to_be_between('tenure', min_value=0, max_value=72)

    # Expectation: 'PhoneService' column should exist and have values 'Yes' or 'No'
    validator.expect_column_to_exist('PhoneService')
    validator.expect_column_values_to_be_in_set('PhoneService', ['Yes', 'No'])

    # Expectation: 'MultipleLines' column should exist and have values 'Yes', 'No', or 'No phone service'
    validator.expect_column_to_exist('MultipleLines')
    validator.expect_column_values_to_be_in_set('MultipleLines', ['Yes', 'No', 'No phone service'])

    # Expectation: 'InternetService' column should exist and have values 'DSL', 'Fiber optic', or 'No'
    validator.expect_column_to_exist('InternetService')
    validator.expect_column_values_to_be_in_set('InternetService', ['DSL', 'Fiber optic', 'No'])

    # Expectation: 'OnlineSecurity' column should exist and have values 'Yes', 'No', or 'No internet service'
    validator.expect_column_to_exist('OnlineSecurity')
    validator.expect_column_values_to_be_in_set('OnlineSecurity', ['Yes', 'No', 'No internet service'])

    # Expectation: 'OnlineBackup' column should exist and have values 'Yes', 'No', or 'No internet service'
    validator.expect_column_to_exist('OnlineBackup')
    validator.expect_column_values_to_be_in_set('OnlineBackup', ['Yes', 'No', 'No internet service'])

    # Expectation: 'DeviceProtection' column should exist and have values 'Yes', 'No', or 'No internet service'
    validator.expect_column_to_exist('DeviceProtection')
    validator.expect_column_values_to_be_in_set('DeviceProtection', ['Yes', 'No', 'No internet service'])

    # Expectation: 'TechSupport' column should exist and have values 'Yes', 'No', or 'No internet service'
    validator.expect_column_to_exist('TechSupport')
    validator.expect_column_values_to_be_in_set('TechSupport', ['Yes', 'No', 'No internet service'])

    # Expectation: 'StreamingTV' column should exist and have values 'Yes', 'No', or 'No internet service'
    validator.expect_column_to_exist('StreamingTV')
    validator.expect_column_values_to_be_in_set('StreamingTV', ['Yes', 'No', 'No internet service'])

    # Expectation: 'StreamingMovies' column should exist and have values 'Yes', 'No', or 'No internet service'
    validator.expect_column_to_exist('StreamingMovies')
    validator.expect_column_values_to_be_in_set('StreamingMovies', ['Yes', 'No', 'No internet service'])

    # Expectation: 'Contract' column should exist and have values 'Month-to-month', 'One year', or 'Two year'
    validator.expect_column_to_exist('Contract')
    validator.expect_column_values_to_be_in_set('Contract', ['Month-to-month', 'One year', 'Two year'])

    # Expectation: 'PaperlessBilling' column should exist and have values 'Yes' or 'No'
    validator.expect_column_to_exist('PaperlessBilling')
    validator.expect_column_values_to_be_in_set('PaperlessBilling', ['Yes', 'No'])

    # Expectation: 'PaymentMethod' column should exist and have values 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', or 'Credit card (automatic)'
    validator.expect_column_to_exist('PaymentMethod')
    validator.expect_column_values_to_be_in_set('PaymentMethod', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

    # Expectation: 'MonthlyCharges' column should exist and have non-negative float values
    validator.expect_column_to_exist('MonthlyCharges')
    validator.expect_column_values_to_be_of_type('MonthlyCharges', 'float64')
    validator.expect_column_values_to_be_between('MonthlyCharges', min_value=0, max_value=250)

    # Expectation: 'TotalCharges' column should exist and have non-negative float values
    validator.expect_column_to_exist('TotalCharges')
    validator.expect_column_values_to_be_of_type('TotalCharges', 'float64')
    validator.expect_column_values_to_be_between('TotalCharges', min_value=0, max_value=15000)

    # Validate the data and collect validation errors
    validator.expect_column_pair_values_A_to_be_greater_than_B('TotalCharges', 'MonthlyCharges', or_equal=True, mostly=0.95) # TotalCharges should be greater than or equal to MonthlyCharges

    print("Data validation completed.")

    validation_results = validator.validate()

    # Extract validation results
    is_valid = _extract_success(validation_results) # Overall success of the validation
    print(f"Validation success: {is_valid}")

    # Extract validation errors
    if isinstance(validation_results, dict):
        results = validation_results.get('results', [])
    else:
        results = getattr(validation_results, 'results', [])

    validation_errors = [
        _extract_expectation_type(result)
        for result in results
        if not _extract_success(result)
    ]
    print(f"Validation errors: {validation_errors}")

    # Summarize validation results
    print("Validation summary:")
    for result in results:
        expectation_type = _extract_expectation_type(result)
        success = _extract_success(result)
        print(f"{expectation_type}: {'Passed' if success else 'Failed'}")

    return is_valid, validation_errors
