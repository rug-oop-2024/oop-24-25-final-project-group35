from typing import List
import pandas as pd
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detects feature types as 'numerical' or 'categorical' for each column in the dataset.
    Assumes no NaN values.

    The function considers a numerical column as categorical if:
    - The number of unique values is less than or equal to a threshold (e.g., 10), or
    - The ratio of unique values to total number of rows is below a certain percentage (e.g., 5%).

    Args:
        dataset: Dataset

    Returns:
        List[Feature]: List of features with their types.
    """
    features = []

    df: pd.DataFrame = dataset.read()
    n_rows = len(df)

    for column in df.columns:
        unique_values = df[column].nunique()
        unique_ratio = unique_values / n_rows

        if pd.api.types.is_numeric_dtype(df[column]):
            if unique_values <= 3:
                feature_type = 'categorical'
            else:
                feature_type = 'numerical'
        else:
            feature_type = 'categorical'

        feature = Feature(name=column, type=feature_type)
        features.append(feature)

    return features

