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
from pytorch_bn_fusion.bn_fusion import fuse_bn_recursively

import torch
import torch.nn as nn

import tensorrt as trt
import torch.onnx
import onnx

class HarDNetModel(nn.Module):
  def __init__(self, num_class, model_path):
    super().__init__()
    #model = get_hardnet(num_layers=num_layers, heads=opt.heads, head_conv=opt.head_conv, trt=True)
    model = get_model({"arch": "hardnet"}, num_class)
    state = convert_state_dict(torch.load(model_path, map_location="cuda")["model_state"])
    model.load_state_dict(state)
 
    model = fuse_bn_recursively(model)
    model.v2_transform(trt=True) 
 
    self.model = model
    
  def forward(self, x):
    #x =  x / 255.0
    #print("img shape after expand",x.shape)
    #print(x)
    out = self.model(x)
    #print("out shape",out.shape)
    out_v = torch.argmax(out, dim=1)
    #print("out shape",out_v.shape)
    return out_v

def onnxConverterIMG(onnxName, modelName, imgName, target_proc_size, dynamicAxis, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 19

    ONNX_GENERATE = True
    if ONNX_GENERATE:
        model = HarDNetModel(n_classes, modelName).to(device)
        model.eval()
        #model.eval()
        #model.to(device)

        # Read image
        img = cv2.imread(imgName)
        img_resized = cv2.resize(img, (target_proc_size[1], target_proc_size[0]))  # uint8 with RGB mode
        img = img_resized.astype(np.float16)

        # norm
        #value_scale = 255
        #mean = [0.406, 0.456, 0.485]
        #mean = [item * value_scale for item in mean]
        #std = [0.225, 0.224, 0.229]
        #std = [item * value_scale for item in std]
        #img = (img - mean) / std
        img = img / 255

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        print("img shape before expand",img.shape)
        img = np.expand_dims(img, 0)

        img = torch.from_numpy(img).float()
        print("img shape after expand",img.shape)
        images = img.to(device)

        #Sample output
        #output_names = ['output', 'output_val']
        output_names = ['output']
        #outputs = model(images)
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
                            output_names = output_names,      # the model's output names
                            dynamic_axes= dynamic_axes,)     # variable lenght axes
                            #example_outputs=outputs)
        else:
            torch.onnx.export(  model,                          # model being run
                            images,                         # model input (or a tuple for multiple inputs)
                            onnxName,                       # where to save the model (can be a file or file-like object)
                            verbose=True, 
                            export_params=True,             # store the trained parameter weights inside the model file
                            opset_version=11,               # the ONNX version to export the model to
                            do_constant_folding=True,       # whether to execute constant folding for optimization
                            input_names = ['input'],        # the model's input names
                            output_names = output_names,      # the model's output names
                            )

        onnxCheck(onnxName)
   
def onnxCheck(onnxName):
    onnx_model = onnx.load(onnxName)
    onnx.checker.check_model(onnx_model)
    print('ONNX Generate Done')

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
        default="FCHardNet_static608_NEW.onnx",
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

    parser.add_argument(
        "--dynamic",
        nargs="?",
        type=bool,
        default=False,
        help="dynamic axis",
    )

    
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    print("Current Working Directory " , os.getcwd())

    onnxName = args.onnx
 

    modelName = args.weight
    imgName = "./pic/aachen_000002_000019_leftImg8bit.png"
    targetSize = (args.width, args.height)
    dynamicAxis = args.dynamic
    
    onnxConverterIMG(onnxName, modelName, imgName, targetSize, dynamicAxis, cfg)
    print("Onnx CONVERT DONE!!!")

    #ToDO: need to check build and converter later...
    #onnxConverter(onnxName, modelName)
    #engineName = "FCHardNet_static608.engine"
    #enginebuild(onnxName, engineName, dynamicAxis, targetSize)