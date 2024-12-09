FROM tensorflow/serving:latest

COPY ./output/serving_model/1733585894 /models/heart-disease-prediction-model/1
ENV MODEL_NAME=heart-disease-prediction-model
ENV PORT=8500

RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
--monitoring_config_file=${MONITORING_CONFIG} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh
