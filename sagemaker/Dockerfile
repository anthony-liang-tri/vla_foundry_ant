ARG AWS_REGION
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-sagemaker

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install -r /opt/ml/code/requirements.txt
RUN pip uninstall flash-attn -y
RUN pip install flash-attn --no-build-isolation --no-cache-dir

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY lbm2/ /opt/ml/code/
COPY secrets.env /opt/ml/code/

# Prevent sagemaker from installing requirements again.
RUN rm /opt/ml/code/requirements.txt