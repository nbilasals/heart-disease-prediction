FROM tensorflow/serving:latest

COPY ./output/serving_model/1733585894 /models/heart-disease-prediction-model/1
ENV MODEL_NAME=heart-disease-prediction-model
