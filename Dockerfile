FROM nvcr.io/nvidia/pytorch:22.12-py3
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

ARG PYTORCH='2.0.1'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu118'

RUN apt -y update && apt upgrade -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.10 python3.10-dev -y
RUN apt install python3.10-distutils -y
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

RUN python3 -m pip uninstall -y torch torchvision torchaudio

# Install latest release PyTorch
RUN python3 -m pip install --no-cache-dir -U torch==$PYTORCH torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python3 -m pip install --no-cache-dir ./transformers[deepspeed-testing]

RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/accelerate@main#egg=accelerate

# Uninstall transformer-engine
RUN python3 -m pip uninstall -y transformer-engine

# Uninstall torch-tensorrt
RUN python3 -m pip uninstall -y torch-tensorrt

# recompile apex
RUN python3 -m pip uninstall -y apex
RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && git checkout 82ee367f3da74b4cd62a1fb47aa9806f0f47b58b && MAX_JOBS=1 python3 -m pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .

# Pre-build latest DeepSpeed
RUN python3 -m pip uninstall -y deepspeed
RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_UTILS=1 python3 -m pip install deepspeed --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1

# Install transformers
RUN cd transformers && python3 setup.py develop

# Upgrade pydantic
RUN python3 -m pip install -U --no-cache-dir "pydantic<2"
RUN python3 -c "from deepspeed.launcher.runner import main"
