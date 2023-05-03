#!/bin/bash

# --------------------------------------------------------------
#
# 	This script starts and manages a docker container to
# 	convert PyTorch .pth models to ONNXRuntime's .ort
#	format.
#
#	USAGE: . convert_model [pytorch_model_file] [N]
#
#	...where N = number of species identified by the model
#
#	Eg: . convert_model model257species.pyt 257
#
#	2pi Software
#	Lachlan Walker
#
# --------------------------------------------------------------

# Strip file extension from model file name
readarray -d . -t strarr <<< "$1"
BASENAME=${strarr}

# Check that exactly 2 cmd line args have been passed
if [[ $# != 2 ]]; then
    echo "Incorrect number of arguments supplied. Please check usage instructions."
    exit 1
fi

# Check existence of specified model file
FILE=$1
if [[ ! -f "$FILE" ]]; then
    echo "Model file not found, please check your input for typos".
    exit 1
fi

# Check for existing Docker image, otherwise build new
if [[ "$(docker images -q convert-mnv2)" == "" ]]; then
	echo "Docker image: convert-mnv2 not found. Building image from Dockerfile..."

	docker build -t convert-mnv2 .
else
	echo "Docker image:convert-mnv2 found."
fi

echo "Spinning up container..."

docker run --name model_converter -dit convert-mnv2

echo "Copying PyTorch model to container..."

docker cp $1 model_converter:/opt/$1
docker cp convert_to_onnx.py model_converter:/opt/convert_to_onnx.py

echo "Copying completed."
echo "Converting model to .onnx..."

# Run .pyt -> .onnx conversion
docker exec model_converter python3 ./opt/convert_to_onnx.py "/opt/$1" $2

# Copy .onnx file back to host to checkfor success
docker cp model_converetr:/opt/${BASENAME}.onnx "${BASENAME}.onnx"

# Catch .pyt -> .onnx failure. 
if [[ ! -f "${BASENAME}.onnx" ]]; then
    echo "ONNX file not found, please check your input for typos."
    exit 1
fi

#docker cp model_converter:/opt/${BASENAME}.log "${BASENAME}.log"

echo "Converting model to .ort..."

# Convert .onnx -> .ort
docker exec model_converter python3 -m onnxruntime.tools.convert_onnx_models_to_ort "/opt/${BASENAME}.onnx" 

echo "Copying .ort file to host machine..."

docker cp model_converter:/opt/${BASENAME}.ort "${BASENAME}.ort"

# Catch .onnx -> .ort, or copy to host, failures
if [[ ! -f "${BASENAME}.ort" ]]; then
    echo "ORT file not found. ONNX -> ORT conversion failed."
    exit 1
fi

echo "Killing container..."

docker stop model_converter > /dev/null
docker rm model_converter > /dev/null

echo "done."
