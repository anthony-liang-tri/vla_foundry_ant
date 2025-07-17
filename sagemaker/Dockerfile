ARG AWS_REGION
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-sagemaker

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# Install from pyproject.toml.
# TODO: Figure out how to install from uv.lock instead.
# The issue with uv.lock is that it seems to rely on the sagemaker tar file
# even if we are installing without that group.
RUN pip install --no-cache-dir uv
COPY pyproject.toml /opt/ml/code/pyproject.toml
RUN uv pip install --python /usr/local/bin/python --no-cache-dir -r /opt/ml/code/pyproject.toml \
    && uv pip uninstall --python /usr/local/bin/python flash-attn \
    && uv pip install --python /usr/local/bin/python flash-attn --no-build-isolation --no-cache-dir \
    && rm -rf /root/.cache/pip \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*
RUN rm /opt/ml/code/pyproject.toml

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY lbm2/ /opt/ml/code/lbm2/
ENV SAGEMAKER_PROGRAM /opt/ml/code/lbm2/main.py