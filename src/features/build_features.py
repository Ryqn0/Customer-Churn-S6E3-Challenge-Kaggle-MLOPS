import pandas as pd

def build_features(df: pd.DataFrame, target_column: str = 'Churn') -> pd.DataFrame:
    """
    Build features for the given DataFrame. Simple example of feature engineering by encoding categorical variables.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing raw data.
    target_column (str): The name of the target column.

    Returns:
    pd.DataFrame: DataFrame with engineered features.
    """

    categorical_cols = df.select_dtypes(include="object").columns.difference([target_column])
    # numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference([target_column])
    
    data_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    return data_encoded