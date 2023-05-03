#!/bin/bash

# --------------------------------------------------------------
#
# 	This script starts and manages a docker container to
# 	convert PyTorch .pth models to ONNXRuntime's .ort
#	format.
#
#	USAGE: . convert_model [pytorch_model_file]
#
#	Eg: . convert_model model257species.pyt
#
#	2pi Software
#	Lachlan Walker
#
# --------------------------------------------------------------

# Strip file extension from model file name
readarray -d . -t strarr <<< "$1"
BASENAME = ${strarr[0]}

# Check that exactly 2 cmd line args have been passed
if [[ $# != 1 ]]; then
    echo "Incorrect number of arguments supplied. Please check usage instructions."
    exit 1
fi

# Check existence of specified model file
FILE=$1
echo "Input model is: " $FILE 
if [[ ! -f "$FILE" ]]; then
    echo "Model file not found, please check your input for typos."
    exit 1
fi

echo "Base model name: " $BASENAME

# Get the number of outputs from the model (N)
FULL_LINE=$(grep -oa "num_classes=[0-9]*" "$FILE")
echo Class spec full line: $FULL_LINE
NUM=$(echo "$FULL_LINE" | grep -Po "[0-9][0-9][0-9]")
echo "Detected N classes = " $NUM

echo "Converting model to .onnx..."

python3 ./convert_to_onnx.py $FILE $NUM

# Catch .pyt -> .onnx failure. 
if [[ ! -f "${BASENAME}.onnx" ]]; then
    echo "ONNX file not found, please check your input for typos."
    exit 1
fi

echo "Converting model to .ort..."

echo "Input onnx model is : ${BASENAME}.onnx"

python3 -m onnxruntime.tools.convert_onnx_models_to_ort "${BASENAME}.onnx" 

# Catch .onnx -> .ort failure
if [[ ! -f "${BASENAME}.ort" ]]; then
    echo "ORT file not found. ONNX -> ORT conversion failed."
    exit 1
fi

# Clean up the excess files generated.
rm ${BASENAME}.onnx
rm ${BASENAME}.required_operators.config
rm ${BASENAME}.required_operators.with_runtime_opt.config

echo "done."
