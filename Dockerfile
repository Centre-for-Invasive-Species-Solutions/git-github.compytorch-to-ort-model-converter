FROM ubuntu:22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip

RUN pip install --upgrade pip

RUN pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install onnx==1.12.1 onnxruntime==1.13.1
