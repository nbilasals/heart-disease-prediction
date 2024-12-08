"""
Preprocessing Script for Heart Disease Prediction Model

This script defines the preprocessing functions used to transform raw input data
into a format suitable for machine learning models. It handles both categorical 
and numerical features, applying transformations such as one-hot encoding for 
categorical data and scaling for numerical data.

Modules:
- TensorFlow
- TensorFlow Transform (tft)

Usage:
- This script is used as part of a data preprocessing pipeline, typically within
  a TensorFlow-based ML workflow to prepare the data for model training.
"""

import tensorflow as tf
import tensorflow_transform as tft

NUMERICAL_FEATURES = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120",
    "EKG results", "Max HR", "Exercise angina", "ST depression",
    "Slope of ST", "Number of vessels fluro", "Thallium"
]

# Label key for classification task
LABEL_KEY = "Heart Disease"

def transformed_name(key):
    """
    Renames the features after transformation to indicate they are transformed.

    Args:
        key (str): The feature name.

    Returns:
        str: The transformed feature name with '_xf' appended to the original key.
    """
    return key + '_xf'


def convert_num_to_one_hot(label_tensor, num_labels=2):
    """
    Converts a numeric label (0 or 1) into a one-hot encoded vector.

    Args:
        label_tensor (tf.Tensor): A tensor containing the label (0 or 1).
        num_labels (int): The number of possible labels (default is 2 for binary classification).

    Returns:
        tf.Tensor: A one-hot encoded tensor.
    """
    # Create a one-hot encoding of the label
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    
    # Ensure the tensor has the correct shape for further processing
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def preprocessing_fn(inputs):
    """
    Preprocesses the input features and applies transformations.

    This function processes both categorical and numerical features, applying
    appropriate transformations to prepare them for model training.

    Args:
        inputs (dict): A dictionary mapping feature names to raw input data.

    Returns:
        dict: A dictionary mapping transformed feature names to transformed data.
    """
    outputs = {}

    # Process numerical features by scaling them to the range [0, 1]
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    # Ensure the label is cast to an integer type
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
