# Pytorch to ORT model conversion 

## Usage 

- Open a codespace on configuration "Single_Container"
- create a directory <MODEL_DIR> at $PWD to contain the model
- copy the .pth file that you want to convert into $PWD/<MODEL_DIR>
- run bash convert_model_noDocker.sh models/<MODEL_DIR>/<MODEL>.pth
- Collect the .ort file, and test and distribute as you wish. 