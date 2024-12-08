FROM tensorflow/serving:latest

COPY ./output/serving_model_dir /models
ENV MODEL_NAME=heart-disease-prediction-model
