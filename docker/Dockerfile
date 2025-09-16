FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    lsb-release \
    gnupg2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    git \
    nano \
    vim \
    build-essential \
    python3-pcl \
    python3-pip \
    python3-dev \
    python3-pip python3-dev \
    tmux \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/anaconda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/anaconda && \
    rm ~/anaconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# Initialize conda for all users
RUN /opt/anaconda/bin/conda init bash

RUN /opt/anaconda/bin/conda install -y jupyter spyder

RUN /opt/anaconda/bin/conda install -y cudatoolkit=11.1 -c conda-forge

RUN /opt/anaconda/bin/pip install nvidia-pyindex && \
    /opt/anaconda/bin/pip install nvidia-tensorrt


WORKDIR /root/deepLearning

EXPOSE 8888

