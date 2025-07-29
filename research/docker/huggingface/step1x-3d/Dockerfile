FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME='/usr/local/cuda'
ARG TORCH_CUDA_ARCH_LIST="8.0"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV PYTHONPATH=${PYTHONPATH}:/root

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /usr/lib/python3.10/dist-packages/blinker* /usr/lib/python3/dist-packages/blinker*
    
WORKDIR /root

RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

RUN git clone --depth 1 --branch main https://github.com/stepfun-ai/Step1X-3D.git
WORKDIR /root/Step1X-3D

ENV FORCE_CUDA=1
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html \
    && pip install --no-cache-dir kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html \
    && cd step1x3d_texture/custom_rasterizer && python setup.py install && cd .. \
    && cd differentiable_renderer && python setup.py install \
    && pip install --no-cache-dir python-magic

COPY huggingface/step1x-3d/ .

CMD ["python", "main.py"]
