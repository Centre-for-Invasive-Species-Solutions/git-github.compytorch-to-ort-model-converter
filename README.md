# pytorch-to-ort-model-converter

This project is used to convert MobileNetV2-architecture neural network models from PyTorch's .pth format to ONNXRuntime's .ort format. The current converter is currently configured for ONNX opset 17.

This project is designed to facilitate model conversion from either local or remote (using GitHub Codespaces) workspaces.

## Codespaces
Execution via a Github Codespace is preferable due to the repeatability of the enviroment. Additionally, for usage on systems which:
* Do not meet the local execution requirements, or
* Produce unexpected results

Codespace execution may be necessary.
### Usage
To run the conversion on codespaces, launch a `docker in docker` codespace from the project splash page. Drag and drop your model-name.pth file into the project directory.
<!-- ?? I don't see that option?  -->
Execute **convert_model.sh** in the codespace terminal by passing in the path to your .pth model, and the number of species identified by the model.

`./convert_model.sh [path to .pth model] [N = number of species identified]`

Eg: `./convert_model.sh model257species.pth 257`

After successful conversion of your model, a .ort file will appear in the Codespace explorer with the same name as the input model. 
To copy the model onto a local machine, right-click and select 'Download' from the context menu.

## Local
Once the docker image has been built, a local instance of this project may facilitate faster conversion by doing away with the need to launch a Codespace.
### Requirements
Local conversion requires:
* Docker
* A terminal capable of executing bash scripts

### Usage
To run the conversion locally, clone the repository to your local machine and copy your model-name.pth file into the project directory.
Execute **convert_model.sh** by passing in the path to your .pth model, and the number of species identified by the model.

`./convert_model.sh [path to .pth model] [number of species identified]`

Eg: `./convert_model.sh model257species.pth 257`
