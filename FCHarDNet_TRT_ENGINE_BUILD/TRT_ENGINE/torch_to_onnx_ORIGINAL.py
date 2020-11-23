import yaml
import torch
import argparse
import timeit
import time
import os
import numpy as np
import cv2
from torch.utils import data

from ptsemseg.models import get_model
from ptsemseg.utils import convert_state_dict


import tensorrt as trt
import torch.onnx
import onnx

def onnxConverterIMG(onnxName, modelName, imgName, target_proc_size, dynamicAxis):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 19

    ONNX_GENERATE = True
    if ONNX_GENERATE:
        # Setup Model
        model = get_model({"arch": "hardnet"}, n_classes)
        model_path = modelName
        

        state = convert_state_dict(torch.load(model_path, map_location=device)["model_state"])
        model.load_state_dict(state)
        model.v2_transform(trt=True) 
        model.eval()
        model.to(device)

        # Read image
        img = cv2.imread(imgName)
        img_resized = cv2.resize(img, (target_proc_size[1], target_proc_size[0]))  # uint8 with RGB mode
        img = img_resized.astype(np.float16)

        # norm
        value_scale = 255
        mean = [0.406, 0.456, 0.485]
        mean = [item * value_scale for item in mean]
        std = [0.225, 0.224, 0.229]
        std = [item * value_scale for item in std]
        img = (img - mean) / std

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        print("img shape before expand",img.shape)
        img = np.expand_dims(img, 0)

        img = torch.from_numpy(img).float()
        print("img shape after expand",img.shape)
        images = img.to(device)

        #Sample output
        outputs = model(images)
        if dynamicAxis:
            dynamic_axes = {"input":{0:"batch_size"}, "output":{0:"batch_size"}}
            torch.onnx.export(  model,                          # model being run
                            images,                         # model input (or a tuple for multiple inputs)
                            onnxName,                       # where to save the model (can be a file or file-like object)
                            verbose=True, 
                            export_params=True,             # store the trained parameter weights inside the model file
                            opset_version=11,               # the ONNX version to export the model to
                            do_constant_folding=True,       # whether to execute constant folding for optimization
                            input_names = ['input'],        # the model's input names
                            output_names = ['output'],      # the model's output names
                            #dynamic_axes= dynamic_axes,     # variable lenght axes
                            example_outputs=outputs)
        else:
            torch.onnx.export(  model,                          # model being run
                            images,                         # model input (or a tuple for multiple inputs)
                            onnxName,                       # where to save the model (can be a file or file-like object)
                            verbose=True, 
                            export_params=True,             # store the trained parameter weights inside the model file
                            opset_version=11,               # the ONNX version to export the model to
                            do_constant_folding=True,       # whether to execute constant folding for optimization
                            input_names = ['input'],        # the model's input names
                            output_names = ['output'],      # the model's output names
                            example_outputs=outputs)

        onnxCheck(onnxName)

def onnxConverter(onnxName, modelName,target_proc_size):        
    n_classes = 19
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Model
    model = get_model({"arch": "hardnet"}, n_classes)
    model_path = modelName
    state = convert_state_dict(torch.load(model_path)["model_state"])
    model.load_state_dict(state)

    model.v2_transform(trt=True) 
    model.to(device)
    
    batch_size = 1
    # Input to the model
    x = torch.randn(batch_size, 3, target_proc_size[0], target_proc_size[1], requires_grad=True).to(device)
    in_x = x / 255.0
    y = model(in_x)

    print("onnx input shape", in_x.shape)
    dynamic_axes = {"input":{0:"batch_size"}, "output":{0:"batch_size"}}
    #torch.onnx.export(model,               # model being run
    #                  x,                         # model input (or a tuple for multiple inputs)
    #                  "FCHardNET.onnx",   # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=11,          # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names = ['input'],   # the model's input names
    #                  output_names = ['output'], # the model's output names
    #                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
    #                                'output' : {0 : 'batch_size'}})
    #torch.onnx.export(model, in_x, "FCHardNet_test3.onnx", verbose=True, opset_version=11, example_outputs=y)

    torch.onnx.export(  model,                          # model being run
                        x,                         # model input (or a tuple for multiple inputs)
                        onnxName,                       # where to save the model (can be a file or file-like object)
                        verbose=True, 
                        export_params=True,             # store the trained parameter weights inside the model file
                        opset_version=11,               # the ONNX version to export the model to
                        do_constant_folding=True,       # whether to execute constant folding for optimization
                        input_names = ['input'],        # the model's input names
                        output_names = ['output'],      # the model's output names
                        #dynamic_axes= dynamic_axes
                        )

    onnxCheck(onnxName)
   
def onnxCheck(onnxName):
    onnx_model = onnx.load(onnxName)
    onnx.checker.check_model(onnx_model)
    print('ONNX Generate Done')

def enginebuild(onnxName, engineName, dynamicAxis, targetSize):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnxName, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
        builder.max_workspace_size = 4<<50
        builder.fp16_mode = True
        builder.max_batch_size = 32
        builder.strict_type_constraints = False        
        print("NETWORK LAYER of Model", network.num_layers)
       
        with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config:
            config.max_workspace_size = 2<<20 # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
            if dynamicAxis:
                profile = builder.create_optimization_profile()
                profile.set_shape("input", (1, 3, targetSize[0], targetSize[1]), (2, 3, targetSize[0], targetSize[1]), (3, 3, targetSize[0], targetSize[1])) 
                config.add_optimization_profile(profile)
            with builder.build_engine(network, config) as engine:
                serialized_engine = engine.serialize()
                with open(engineName, 'wb') as f: f.write(serialized_engine)
            print("ENGINE Detail:", engine)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/hardnet.yml",
        help="Config file to be used",
    )

    parser.add_argument(
        "--onnx",
        nargs="?",
        type=str,
        default="FCHardNet_static608.onnx",
        help="target output onnx file name",
    )

    parser.add_argument(
        "--width",
        nargs="?",
        type=int,
        default=608,
        help="target input width",
    )

    parser.add_argument(
        "--height",
        nargs="?",
        type=int,
        default=608,
        help="target input height",
    )

    parser.add_argument(
        "--weight",
        nargs="?",
        type=str,
        default="./weights/hardnet70_cityscapes_model.pkl",
        help="weight file name",
    )

    
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    print("Current Working Directory " , os.getcwd())

    onnxName = args.onnx
 

    modelName = args.weight
    imgName = "./pic/aachen_000002_000019_leftImg8bit.png"
    targetSize = (args.width, args.height)
    dynamicAxis = False
    
    onnxConverterIMG(onnxName, modelName, imgName, targetSize, dynamicAxis)
    print("Onnx CONVERT DONE!!!")

    #ToDO: need to check build and converter later...
    #onnxConverter(onnxName, modelName)
    #engineName = "FCHardNet_static608.engine"
    #enginebuild(onnxName, engineName, dynamicAxis, targetSize)