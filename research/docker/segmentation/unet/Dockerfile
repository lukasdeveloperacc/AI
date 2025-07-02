FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

RUN pip install --upgrade && pip install --no-cache-dir hydra onnx
RUN pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
RUN pip install --upgrade tensorrt-cu11 pycuda

COPY common .
COPY segmentation/unet/ segmentation/unet

ENV GIT_PYTHON_REFRESH="quiet"
