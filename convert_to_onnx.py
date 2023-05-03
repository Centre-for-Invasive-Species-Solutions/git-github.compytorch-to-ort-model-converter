"""
This file is part of the weedscan project.
It is used to convert mobilenet-v2 models from PyTorch's .pth format to .onnx

2Pi Software
Lachlan Walker 2023
"""

from torchvision import models
from torch import nn
import logging, argparse
import torch
import onnx, onnxruntime
from collections import OrderedDict
import numpy as np

## Usage
# python convert_to_onnx.py "./model257species_epoch_250.pth" 257

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', '-t', action='store_true')
    parser.add_argument('modelpath', type=str, help='path to pytorch model')
    parser.add_argument('n_species', type=int, help='number of species classified by the model')
    args = parser.parse_args()

    modelname = args.modelpath.split(".")[0]
    logfile = modelname + ".log"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(filename=logfile, level=logging.DEBUG, format='%(levelname)s - %(message)s')


    # Create an instance of the mobilenet_v2 NN which will be the backbone of the model
    mnv2 = models.mobilenet_v2()

    # Modify the number of output features in the classifier head
    classifier = list(mnv2.classifier.children())
    mnv2.classifier = nn.Sequential(*classifier[:-1])
    mnv2.classifier.add_module('1', nn.Linear(in_features=1280, out_features=args.n_species, bias=True))
    # Re-attach a softmax activation layer to the classifier head
    mnv2.classifier.add_module('2', nn.Softmax(dim=1))

    logging.info('Custom MobileNet V2 instance created.')

    # Load the state dictionary of model parameters from file
    loaded_model_state = torch.load(args.modelpath, map_location=torch.device('cpu'))
    loaded_model_state = loaded_model_state['state_dict']

    logging.info('Model paramters loaded to MEMORY successfully.')

    # Create a list of parameter sets as per-layer tensors
    loaded_model_layers = list(loaded_model_state.keys())
    params = []
    for layer in loaded_model_layers:
        params.append(loaded_model_state[layer])

    # Create a new state dictionary for the generated model
    # and attach the loaded parameters to the new layer names
    new_layer_names = list(mnv2.state_dict().keys())
    new_state = OrderedDict()
    for i in range(len(new_layer_names)):
        new_state[new_layer_names[i]] = params[i]

    # Load the new state dictionary into the generated model
    mnv2.load_state_dict(new_state)

    logging.info('Model paramters loaded to MODEL successfully.')

    # Create random input tensor for model tracing
    batch_size = 1
    m_dim = 224
    n_dim = 224
    x = torch.randn(batch_size, 3, m_dim, n_dim)

    # Set the model to evaluation/testing mode
    mnv2.eval()

    # Set output .onnx model filename	
    onnx_model_name = modelname + ".onnx"

    # Convert model to .onnx format via operation tracing
    torch.onnx.export(mnv2, 				# Model to convert
        x, 									# Dummy input tensor for trace
        onnx_model_name, 					# Output model name
        export_params=True, 				# Store trained parameter weights
        opset_version=17,					# See https://onnxruntime.ai/docs/reference/compatibility.html for more info
        do_constant_folding=True,			# Constant folding for optimization
        input_names = ['input'],			# Model's input tensor name
        output_names = ['output'],			# Model's output tensor name
        dynamic_axes = {'input' : {0 : 'batch_size', 2 : 'm_dim', 3 : 'n_dim'}, 
        'output' : {0 : 'batch_size'}})		# Specify input tensor axes whose sizes may change when running the model

    logging.info('Model traced to .onnx successfully.')

    # OPTIONAL: test output .onnx model results against the input .pth model results
    if args.test:
        logging.info('Testing converted model...')

        # Output of the pytorch model for the dummy input tensor
        torch_out = mnv2(x)

        logging.info('Pytorch model evaluation result has shape: ', torch_out.shape)

        # Load the .onnx model and verify that it has a executable graph
        onnx_model = onnx.load(onnx_model_name)
        onnx.checker.check_model(onnx_model)

        logging.info('Loaded and checked ONNX model graph.')

        # Initialise onnx runtime session
        ort_session = onnxruntime.InferenceSession(onnx_model_name)

        logging.info('ONNXRuntime session initialised.')

        # Handles requirements of model tensors to have gradients calculated
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)

        logging.info('ONNXRuntime model evaluation result has shape: ', ort_outs.shape)

        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime.")
        logging.info('Model results match.')
