FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive # for ubuntu 20.04

RUN apt-get update  -y \ 
  && apt-get install -y git cmake make gnupg build-essential software-properties-common gdb unzip\ 
  && add-apt-repository ppa:deadsnakes/ppa \
  && apt install --no-install-recommends -y python3.8 python3-pip python3-setuptools python3-distutils \
  && apt-get install -y libgl1-mesa-glx python3-opencv\
  && apt-get install -y nano ffmpeg libsm6 libxext6 \
  && apt clean \
  && rm -rf /var/lib/apt/lists/* \
  && pip3 install --upgrade pip opencv-python opencv-python-headless 

RUN git clone https://github.com/SrCarlos01/yolov7.git \
    && mkdir -p yolov7/weights \
    && mkdir datasets\
    && pip install Cython==0.29.23 numpy==1.20.1

WORKDIR /yolov7
