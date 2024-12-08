"""
TFX Pipeline for building, training, evaluating, and deploying a machine learning model.

This script defines the `init_components` function, which initializes the necessary components 
for a TFX pipeline. These components handle data ingestion, validation, transformation, model 
training, evaluation, and deployment to a serving directory.

Main Components:
- CsvExampleGen: Reads input data from CSV files.
- StatisticsGen: Generates data statistics.
- SchemaGen: Creates a data schema.
- ExampleValidator: Validates input data against the schema.
- Transform: Preprocesses/transforms data for model training.
- Trainer: Trains a model using the transformed data.
- Evaluator: Evaluates the model against a baseline with defined metrics.
- Pusher: Deploys the trained model to a specified directory.

Dependencies:
- TensorFlow
- TensorFlow Model Analysis (TFMA)
- TFX
"""

import os
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)


def init_components(
    data_dir: str,
    transform_module: str,
    training_module: str,
    training_steps: int,
    eval_steps: int,
    serving_model_dir: str
):
    """
    Initializes TFX pipeline components for building, training, and evaluating an ML model.

    Args:
        data_dir (str): Path to the directory containing input CSV data.
        transform_module (str): Path to the file containing the data transformation module.
        training_module (str): Path to the file containing the training module.
        training_steps (int): Number of training steps to execute.
        eval_steps (int): Number of evaluation steps to execute.
        serving_model_dir (str): Path to the directory where the trained model will be deployed.

    Returns:
        Tuple: A tuple of initialized TFX components required to run the pipeline.
    """

    # Component 1: CsvExampleGen for ingesting input CSV data
    example_gen = CsvExampleGen(
        input_base=data_dir,
        output_config=example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                # Splitting data into 80% training and 20% evaluation
                example_gen_pb2.SplitConfig.Split(
                    name='train', hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2),
            ])
        )
    )

    # Component 2: StatisticsGen for generating statistics on the input data
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples']
    )

    # Component 3: SchemaGen for inferring the schema of the input data
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"]
    )

    # Component 4: ExampleValidator for validating the input data against the schema
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # Component 5: Transform for preprocessing and feature engineering
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        # Absolute path to the transform module
        module_file=os.path.abspath(transform_module)
    )

    # Component 6: Trainer for training the ML model
    trainer = Trainer(
        # Absolute path to the training module
        module_file=os.path.abspath(training_module),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],  # Train on the training split
            num_steps=training_steps  # Specify the number of training steps
        ),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],  # Evaluate on the evaluation split
            num_steps=eval_steps  # Specify the number of evaluation steps
        )
    )

    # Component 7: Resolver to retrieve the latest blessed model as the baseline
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    # Component 8: Evaluator for evaluating the trained model with defined metrics
    slicing_specs = [
        tfma.SlicingSpec(),  # Default slicing
        # Custom slicing by features
        tfma.SlicingSpec(feature_keys=["Sex", "Exercise angina"])
    ]
    metrics_specs = [
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name="Precision"),
            tfma.MetricConfig(class_name="Recall"),
            tfma.MetricConfig(class_name="ExampleCount"),
            tfma.MetricConfig(
                class_name='BinaryAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': 0.5}),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value': 0.0001}
                    )
                )
            )
        ])
    ]
    eval_config = tfma.EvalConfig(
        # Label key for the target variable
        model_specs=[tfma.ModelSpec(label_key='Heart Disease')],
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs

    )
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )

    # Check the evaluator output
    evaluator_output = evaluator.outputs['blessing']
    print("Evaluator blessing output:", evaluator_output)

    # Component 9: Pusher for deploying the trained model to the serving directory
    pusher = Pusher(
        model=trainer.outputs["model"],
        # Ensure model is blessed before pushing
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir  # Path to save the model
            )
        )
    )
    # Check the destination directory for the pushed model
    print("Model is being pushed to:", serving_model_dir)

    # Return all initialized components as a tuple
    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )

    return components
