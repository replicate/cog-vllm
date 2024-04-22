FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends -yy build-essential libpython3.10-dev python3.10 curl git
RUN curl -L https://bootstrap.pypa.io/get-pip.py|python3.10
RUN pip install git+https://github.com/microsoft/DeepSpeed.git
RUN pip install git+https://github.com/Snowflake-Labs/vllm.git@yak-staging
RUN pip install git+https://github.com/Snowflake-Labs/transformers.git@yak-staging
