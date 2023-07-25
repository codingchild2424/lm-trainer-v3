# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-12.html#rel-22-12
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

RUN apt update && sudo apt upgrade -y

RUN apt install software-properties-common -y

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt install python3.10 python3.10-dev

RUN alias python='python3.10'

RUN apt install python3.10-distutils

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN apt install git

RUN apt-get install cmake

RUN git clone https://github.com/microsoft/DeepSpeed/

RUN cd DeepSpeed

RUN rm -rf build

RUN TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log