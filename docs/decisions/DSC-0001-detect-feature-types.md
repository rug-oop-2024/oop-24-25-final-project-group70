# DSC-0001: Implement `detect_feature_types` Function
# Date: 2024-11-04
# Decision: Implement `detect_feature_types` to classify feature types as numerical or categorical
# Status: Accepted
# Motivation: To automatically detect and classify feature types in a dataset, simplifying data preprocessing for machine learning
# Reason: Identifying feature types allows for appropriate preprocessing and model compatibility, supporting automated machine learning workflows
# Limitations: Only detects numerical and categorical features, assumes no missing values, may misclassify with complex data
# Alternatives: Extend feature detection to include other types (e.g., datetime, boolean); use custom rules for categorical detection based on unique value thresholds
