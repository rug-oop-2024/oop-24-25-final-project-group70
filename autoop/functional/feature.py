from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detects the types of features in a dataset.

    Assumption: only categorical and numerical features, and no NaN values.

    Args:
        dataset (Dataset): The dataset object to inspect.

    Returns:
        List[Feature]: List of features with their types
        (categorical or numerical).
    """

    # We first load the dataset into a DataFrame.
    df = dataset.read()

    # We initialize the list to store Feature objects.
    features = []

    # Iterate through each column to determine the feature type.
    for column_name in df.columns:
        # Check if column is numerical or categorical.
        if pd.api.types.is_numeric_dtype(df[column_name]):
            feature_type = "numerical"
        else:
            feature_type = "categorical"

        # Create a Feature object and append it to the list.
        feature = Feature(name=column_name, type=feature_type)
        features.append(feature)

    return features
