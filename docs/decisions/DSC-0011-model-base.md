# DSC-0011: Implement `Model` Base Class for Machine Learning Models
# Date: 2024-11-09
# Decision: Create a `Model` base class with methods for fitting, predicting, and managing model parameters, supporting both strict parameters and hyperparameters.
# Status: Accepted
# Motivation: To provide a standardized structure for defining machine learning models, making it easy to save and load model states, and manage both training and prediction parameters.
# Reason: A base class for models allows for consistent implementation across various machine learning models, supporting extensibility and reusability. The ability to save and load model parameters enhances model reproducibility and supports workflow automation.
# Limitations: The base class does not define specific model behavior for fitting and predicting, requiring subclasses to implement these methods. Saving and loading assumes that the `Artifact` class is capable of handling the model’s parameters.
# Alternatives: Implement separate model classes without a common base class, but this would lead to duplicated code and inconsistencies in parameter management and saving/loading functionality.
