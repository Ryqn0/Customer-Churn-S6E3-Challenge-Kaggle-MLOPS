import pandas as pd

def preprocess_data(data: pd.DataFrame, target_column: str = 'Churn') -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and encoding categorical variables.

    Parameters:
    data (pd.DataFrame): The input DataFrame to preprocess.

    Returns:
    pd.DataFrame: A preprocessed DataFrame ready for analysis or modeling.
    """

    print("Preprocessing data...")

    print(f"Initial data shape: {data.shape}")

    # Handle dupplicate rows
    data = data.drop_duplicates()

    print(f"Data shape after dropping duplicates: {data.shape}")

    print("Handling missing values and encoding categorical variables...")

    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].fillna(data[column].mode()[0]) # Fill missing categorical values with the mode
        elif data[column].dtype == 'bool':
            data[column] = data[column].fillna(data[column].mode()[0]) # Fill missing boolean values with the mode
        else:
            if column == "SeniorCitizen":
                data[column] = data[column].fillna(data[column].mode()[0]) # Fill missing values in SeniorCitizen with the mode
                data[column] = data[column].map({0: 'No', 1: 'Yes'}) # Map 0 and 1 to 'No' and 'Yes' respectively
            else:
                data[column] = data[column].fillna(data[column].median()) # Fill missing numerical values with the median
    
    print("Missing values handled.")

    if target_column in data.columns:
        data[target_column] = data[target_column].map({'No': 0, 'Yes': 1}) # Map target variable to binary values

    print("Target variable encoded.")

    #categorical_cols = data.select_dtypes(include="object").columns.difference([target_column])
    
    # Encode categorical variables using one-hot encoding
    #data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=False)
    
    print("Preprocessing completed.")

    return data