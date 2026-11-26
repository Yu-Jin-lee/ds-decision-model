# syntax=docker/dockerfile:1
# Image of one model's SageMaker real-time endpoint (method `endpoint`, bench/endpoint/).
# Built by bench/endpoint/build_image.sh with train/ as the context:
#
#   docker build --platform linux/amd64 -f bench/endpoint/Dockerfile \
#     --build-arg BASE_IMAGE=<training image> --build-arg MODEL=kev-4b -t <ECR repo>:kev-4b-<commit> .
#
# Same base image and the same installs as the serving and load-test jobs (sagemaker/entry.py install()),
# done at build time. Model weights are not in the image: the container downloads them when it starts
# (bench/endpoint/serve.py), then serves on port 8080 (/ping, /invocations).
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MODEL
ENV PYTHONUNBUFFERED=1 \
    KEV_DIR=/opt/kev \
    VLLM_VENV=/opt/vllm-venv \
    DECISION_MODEL_NAME=${MODEL}

COPY . /opt/train
RUN python /opt/train/sagemaker/entry.py --model ${MODEL} --serving 1 --install-only

EXPOSE 8080
# SageMaker starts the container as `docker run <image> serve`; serve.py ignores the argument.
ENTRYPOINT ["python", "/opt/train/bench/endpoint/serve.py"]
