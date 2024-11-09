# DSC-0003: Implement `Feature` Class for Dataset Feature Representation
# Date: 2024-11-08
# Decision: Create `Feature` class to represent individual dataset features, including type and statistical attributes
# Status: Accepted
# Motivation: To provide a standardized structure for managing features within datasets, including classification by type and basic statistics
# Reason: The `Feature` class enables easy identification and handling of feature-specific properties, supporting feature engineering and preprocessing in machine learning workflows
# Limitations: Only supports `numerical` and `categorical` types; statistics are limited to mean for numerical features and unique value count for categorical features
# Alternatives: Extend to additional feature types (e.g., datetime, boolean) and add further statistical methods (e.g., median, standard deviation) for a more comprehensive feature analysis
