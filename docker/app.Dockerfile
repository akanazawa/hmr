# syntax = docker/dockerfile:experimental
FROM tensorflow/tensorflow:1.12.3-py3

# Remove cuda sources list since it's key is not valid anymore, update and install a few dependencies
# RUN rm /etc/apt/sources.list.d/cuda.list && \
#     rm /etc/apt/sources.list.d/nvidia-ml.list && \
#     apt update && apt install -y libcairo2-dev libgl1 freeglut3-dev xvfb

RUN apt update && apt install -y libglu1-mesa-dev libosmesa6-dev

# RUN mkdir -p /root/.keras/models
# ADD https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5 \
#     /root/.keras/models/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5

# # Install python environment
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Setup bashrc
COPY ./docker/bashrc /root/.bashrc

# Setup PYTHONPATH
ENV PYTHONPATH=.

# Tensorflow keeps on using deprecated APIs ^^
ENV PYTHONWARNINGS="ignore::DeprecationWarning:tensorflow"

# # Set up working directory
WORKDIR /app
