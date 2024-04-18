#syntax=docker/dockerfile:1.4
FROM python:3.11 as deps
COPY .cog/tmp/build413441631/cog-0.0.1.dev-py3-none-any.whl /tmp/cog-0.0.1.dev-py3-none-any.whl
RUN --mount=type=cache,target=/root/.cache/pip pip install -t /dep /tmp/cog-0.0.1.dev-py3-none-any.whl
COPY .cog/tmp/build413441631/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -t /dep -r /tmp/requirements.txt
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin
ENV NVIDIA_DRIVER_CAPABILITIES=all
RUN --mount=type=cache,target=/var/cache/apt set -eux; \
apt-get update -qq; \
apt-get install -qqy --no-install-recommends curl; \
rm -rf /var/lib/apt/lists/*; \
TINI_VERSION=v0.19.0; \
TINI_ARCH="$(dpkg --print-architecture)"; \
curl -sSL -o /sbin/tini "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TINI_ARCH}"; \
chmod +x /sbin/tini
ENTRYPOINT ["/sbin/tini", "--"]
ENV PATH="/root/.pyenv/shims:/root/.pyenv/bin:$PATH"
RUN --mount=type=cache,target=/var/cache/apt apt-get update -qq && apt-get install -qqy --no-install-recommends \
	make \
	build-essential \
	libssl-dev \
	zlib1g-dev \
	libbz2-dev \
	libreadline-dev \
	libsqlite3-dev \
	wget \
	curl \
	llvm \
	libncurses5-dev \
	libncursesw5-dev \
	xz-utils \
	tk-dev \
	libffi-dev \
	liblzma-dev \
	git \
	ca-certificates \
	&& rm -rf /var/lib/apt/lists/*
RUN curl -s -S -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash && \
	git clone https://github.com/momo-lab/pyenv-install-latest.git "$(pyenv root)"/plugins/pyenv-install-latest && \
	pyenv install-latest "3.11" && \
	pyenv global $(pyenv install-latest --print "3.11") && \
	pip install "wheel<1"
RUN --mount=type=bind,from=deps,source=/dep,target=/dep cp -rf /dep/* $(pyenv prefix)/lib/python*/site-packages || true
RUN pip freeze |tee list1
RUN pip install https://r2.drysys.workers.dev/tmp/cog-0.10.0a6-py3-none-any.whl
RUN pip freeze |tee list2
# RUN CUDA_HOME=/usr/local/cuda pip install --ignore-installed git+https://github.com/microsoft/DeepSpeed.git
# RUN git clone -b yak-staging https://github.com/Snowflake-Labs/vllm.git && cd vllm && CUDA_SELECT_NVCC_ARCH_FLAGS="8.0;8.6;8.9;9.0" TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" CUDA_HOME=/usr/local/cuda mount=type=cache,target=/root/.cache/pip pip install --ignore-installed -e .
# RUN git clone -b yak-staging https://github.com/Snowflake-Labs/transformers.git && cd transformers && CUDA_SELECT_NVCC_ARCH_FLAGS="8.0;8.6;8.9;9.0" TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" CUDA_HOME=/usr/local/cuda mount=type=cache,target=/root/.cache/pip pip install --ignore-installed -e .
RUN pip install -U pydantic==2.0.0
RUN pip install -U fastapi
RUN pip freeze |tee list3
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.5.4/pget_linux_x86_64" && chmod +x /usr/local/bin/pget
RUN bash -c 'ln -s /usr/local/lib/python3.11/site-packages/torch/lib/lib{nv,cu}* /usr/lib'
RUN pip install scipy==1.11.4 sentencepiece==0.1.99 protobuf==4.23.4 python-dotenv
RUN ln -sf $(which echo) $(which pip)
WORKDIR /src
EXPOSE 5000
CMD ["python", "-m", "cog.server.http"]
COPY . /src
