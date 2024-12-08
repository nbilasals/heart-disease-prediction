"""
Training Module for Heart Disease Prediction Model

This module defines functions to create, train, and save a machine learning model 
for predicting heart disease based on input features. It leverages TensorFlow 
and Keras to build and train the model, while using TensorFlow Transform (tft) 
to preprocess the data.

Modules:
- TensorFlow
- Keras
- TensorFlow Transform (tft)
- TFRecord for data input/output

Usage:
- This script is typically used in a TensorFlow Extended (TFX) pipeline for model 
  training and deployment.
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
import tensorflow_transform as tft
from transform import (
    NUMERICAL_FEATURES,
    LABEL_KEY,
    transformed_name,
)


def get_model(show_summary=True):
    """
    Defines and returns a Keras model for heart disease prediction.
    
    The model consists of both categorical and numerical inputs, processes them 
    through a series of dense layers, and outputs a binary classification (sigmoid).
    
    Args:
        show_summary (bool): If True, prints the summary of the model architecture.
        
    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    
    # Prepare the list of input features
    input_features = []
    
    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )
    
    # Combine all inputs into a single tensor
    concatenate = tf.keras.layers.concatenate(input_features)
    
    # Add deep dense layers
    deep = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=l2(0.01))(concatenate)  # Regularization added
    deep = tf.keras.layers.Dropout(0.5)(deep)  # Dropout layer with 50% dropout rate
    deep = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=l2(0.01))(deep)  # Regularization added
    deep = tf.keras.layers.Dropout(0.5)(deep)  # Dropout layer with 50% dropout rate

    
    # Output layer (binary classification)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(deep)
    
    # Create the model
    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    
    if show_summary:
        model.summary()
    
    return model


def gzip_reader_fn(filenames):
    """
    Loads compressed TFRecord data files.

    Args:
        filenames (str or list): File paths to the TFRecord files.

    Returns:
        tf.data.TFRecordDataset: The dataset object for reading TFRecord files.
    """
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def get_serve_tf_examples_fn(model, tf_transform_output):
    """
    Returns a function that parses a serialized tf.Example for serving.
    
    This function applies the feature transformation and then runs the model 
    on the transformed features.
    
    Args:
        model (tf.keras.Model): The trained model to be served.
        tf_transform_output (tft.TFTransformOutput): The output of the transform module.

    Returns:
        function: A function that takes serialized tf.Example and returns the model's predictions.
    """
    
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """
        Parses serialized tf.Example and returns model predictions.
        
        Args:
            serialized_tf_examples (tf.Tensor): Serialized tf.Example input data.
            
        Returns:
            dict: A dictionary containing the model's predictions as the 'outputs' key.
        """
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        
        # Transform the features
        transformed_features = model.tft_layer(parsed_features)
        
        # Get model predictions
        outputs = model(transformed_features)
        return {"outputs": outputs}
    
    return serve_tf_examples_fn


def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """
    Generates batched features and labels for training or evaluation.
    
    This function prepares the dataset using the transformed features and 
    applies batching to prepare the data for training or evaluation.

    Args:
        file_pattern (str): The file pattern for the input tfrecord files.
        tf_transform_output (tft.TFTransformOutput): The output of the transform module.
        batch_size (int): The batch size for dataset batching.

    Returns:
        tf.data.Dataset: A dataset containing features and labels.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    
    # Create the batched dataset from TFRecord files
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )
    
    return dataset


def run_fn(fn_args):
    """
    Trains the heart disease prediction model based on input arguments.
    
    This function will load the training and evaluation datasets, create the 
    model, and train it using the specified parameters.
    
    Args:
        fn_args: Arguments for training the model, which include:
            - train_files: Path to the training data.
            - eval_files: Path to the evaluation data.
            - transform_output: Path to the transformation output.
            - serving_model_dir: Directory where the trained model should be saved.
            - train_steps: Number of steps to run training.
            - eval_steps: Number of steps to run evaluation.
    
    Returns:
        None
    """
    
    # Load transformation output and datasets
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 64)
    
    # Create the model
    model = get_model()
    
    # Set up the TensorBoard callback
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )
    
    # Train the model
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
        epochs=10
    )
    
    # Define model signatures for serving
    signatures = {
        "serving_default": get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    
    # Save the model
    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=signatures
    )
    
    # Save the model architecture plot
    plot_model(
        model, 
        to_file='/content/a443-cc-pipeline/images/model_plot.png', 
        show_shapes=True, 
        show_layer_names=True
    )
