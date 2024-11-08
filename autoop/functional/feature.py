from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    # Initialize an empty list to store features
    features = []

    # Go through each column in the dataset's DataFrame
    for column_name in dataset.data.columns:
        # Determine the type of feature based on data type
        if pd.api.types.is_numeric_dtype(dataset.data[column_name]):
            feature_type = "numerical"
        else:
            feature_type = "categorical"

        # Create a Feature object and add it to the features list
        feature = Feature(name=column_name, type=feature_type)
        features.append(feature)

    return features
