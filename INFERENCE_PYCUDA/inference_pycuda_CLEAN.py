import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda 
import time
import os
import cv2
import imageio
import torch
import argparse

import matplotlib.pyplot as plt
import matplotlib.image as mpimg 

from torch.utils import data
from loader import get_loader
from metrics import runningScore

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def alloc_buf(engine):
    # host cpu mem
    h_in_size   = trt.volume(engine.get_binding_shape(0))
    h_out_size  = trt.volume(engine.get_binding_shape(1))
    h_in_dtype  = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    in_cpu      = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    out_cpu     = cuda.pagelocked_empty(h_out_size, h_out_dtype)
    
    # allocate gpu mem
    in_gpu      = cuda.mem_alloc(in_cpu.nbytes)
    out_gpu     = cuda.mem_alloc(out_cpu.nbytes)
    stream      = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream

def read_img(img_path, size):
    #print("Read Input Image from : {}".format(img_path))
    img         = cv2.imread(img_path)
    img_resized = cv2.resize(img, (size[1], size[0]))  # uint8 with RGB mode
    img         = img_resized.astype(np.float32)

    # norm
    # value_scale = 255
    # mean        = [0.406, 0.456, 0.485]
    # mean        = [item * value_scale for item in mean]
    # std         = [0.225, 0.224, 0.229]
    # std         = [item * value_scale for item in std]
    img         = img / 255.0

    # NHWC -> NCHW
    img         = img.transpose(2, 0, 1)
    img         = np.ascontiguousarray(img)
    img         = img[np.newaxis,:]

    images      = img.astype(np.float32)
    return images

def draw_results(seg_map):
    ## bgr
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [232, 35, 244],
        [70, 70, 70],
        [156, 102, 102],
        [153, 153, 190],
        [153, 153, 153],
        [30, 170, 250],
        [0, 220, 220],
        [35, 142, 107],
        [152, 251, 152],
        [180, 130, 0],
        [60, 20, 220],
        [0, 0, 255],
        [142, 0, 0],
        [70, 0, 0],
        [100, 60, 0],
        [100, 80, 0],
        [230, 0, 0],
        [32, 11, 119],
    ]
    seg_image = label_to_color_image(seg_map, colors).astype(np.uint8)
    # seg_image = seg_image.astype(np.uint8)
    return seg_image

def label_to_color_image(seg_map, colors):
    if seg_map.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = np.array(colors)

    if np.max(seg_map) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[seg_map]

def postprocess_images(images, predIMG, blend_img=False):
    # outputs =torch.tensor(predIMG, device="cpu").float()
    # pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    # pred = color_map(predIMG)
    # decoded = decode_segmap(pred)
    decoded = draw_results(predIMG)

    if blend_img:
        img_input = np.squeeze(images,axis=0) * 255.
        img_input = img_input.transpose(1, 2, 0)
        decoded = img_input * 0.4 + decoded * 0.6

    # decoded = decoded.astype(np.uint8)
    return decoded

def save_images2(images, predIMG, fname="test_engine.png", dir_path_id="./img_out_id/", dir_path_img="./img_out_img/", engine_img=False):
    predIMG     = np.squeeze(predIMG, axis = 0)
    decoded     = decode_segmap(predIMG)

    #img_input = np.squeeze(images/255.0,axis = 0)

    #img_input =  img_input.transpose(1, 2, 0)
    #blend     =  img_input * 0.2 + decoded * 0.8

    imageio.imwrite(dir_path_img + fname + "_predSeg_pycuda.jpg", decoded)

def save_images(images, predIMG, fname="test_engine.png", dir_path_id="./img_out_id/", dir_path_img="./img_out_img/", engine_img=False):
    save_segment_map = True
    save_rgb         = True
    
    if save_segment_map:
        if engine_img:
            outputs = torch.tensor(predIMG, device = "cpu").float()
            pred    = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis = 0)
        else:
            pred    = np.squeeze(predIMG.data.max(1)[1].cpu().numpy(), axis = 0)
        
        decoded     = decode_segmap_id(pred)
        dir         = dir_path_id
        if not os.path.exists(dir):
          os.makedirs(dir, mode = 0o777)

        imageio.imwrite(dir + fname + "_segID_pycuda.jpg", decoded)

        if save_rgb:
            decoded = decode_segmap(pred)
            if engine_img:
                img_input = np.squeeze(images,axis = 0)
            else:
                img_input = np.squeeze(images.cpu().numpy(),axis = 0)
            
            img_input =  img_input.transpose(1, 2, 0)
            blend     =  img_input * 0.2 + decoded * 0.8
            fname_new =  fname
            fname_new =  fname_new[:-4]
            fname_new += '_predSeg_pycuda.jpg'
            dir       =  dir_path_img
            if not os.path.exists(dir):
              os.makedirs(dir, mode=0o777)
            imageio.imwrite(dir + fname_new, blend)

def decode_segmap_id(temp):
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]
    n_classes = 19

    ids = np.zeros((temp.shape[0], temp.shape[1]),dtype=np.uint8)
    for l in range(0, n_classes):
        ids[temp == l] = valid_classes[l]
    return ids

def decode_segmap(temp):
    colors = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
              [0, 130, 180],  [220, 20, 60],  [255, 0, 0],  [0, 0, 142],     [0, 0, 70],      [0, 60, 100],    [0, 80, 100],   [0, 0, 230],   [119, 11, 32], ]

    label_colours = dict(zip(range(19), colors))

    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    n_classes = 19
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)
        
        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1
        
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument("--engine", nargs="?", type=str, default="../bin/FCHardNet_static608_fp16.engine", help="engine file name",)
    parser.add_argument("--width", nargs="?", type=int, default=608, help="target input width",)
    parser.add_argument("--height", nargs="?", type=int, default=608, help="target input height", )
    parser.add_argument("--input", nargs="?", type=str, default="../img/berlin_000000_000019_leftImg8bit.png", help="input file name or input directory depends on inputTestType", )
    parser.add_argument("--inputTestType", nargs="?", type=str, default="single", help="input test TYPE: single / dir", )
    parser.add_argument("--showResult", nargs="?", type=bool, default=True, help="Show segmentation result, result file saved in /pred/ folder",)
    parser.add_argument("--classes", nargs="?", type=int, default=19, help="weight file name",)

    args = parser.parse_args()

    if args.inputTestType == "dir":
        inference_with_DIR = True
        inference_with_CITY = False
        input_dir_path = args.input
    elif args.inputTestType == "CITY":
        inference_with_CITY = True
    else:
        inference_with_CITY = False
        inference_with_DIR = False
        single_test_img_path = args.input

    save_img = args.showResult

    engine_NAME = args.engine
    
    #n_classes = args.classes #Segmented output class number
    
    proc_size = [args.width, args.height]
    
    total_time = 0; i = 0; s = 0

    input_dir_path = "/workspace/dataset/KETIDB"

    # Setup Dataloader
    data_loader = get_loader("cityscapes")
    data_path = "/workspace/dataset/CITYSCAPES"

    loader = data_loader(data_path, split="val", is_transform=True, img_size=(608,608),)
    loader_gt = data_loader(data_path, split="val", is_transform=True,  img_size=(1024,2048),)
    
    valloader = data.DataLoader(loader, batch_size=1, num_workers=1)
    valloader_gt = data.DataLoader(loader, batch_size=1, num_workers=1)

    n_classes = loader.n_classes

    running_metrics = runningScore(n_classes)

    ### Read the serialized ICudaEngine
    with open(engine_NAME, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        #print("Engine loaded!!!")
    #print(engine)

    ### create buffer
    #host_inputs  = []
    #cuda_inputs  = []
    #host_outputs = []
    #cuda_outputs = []
    #bindings = []
    #stream = cuda.Stream()

    #for binding in engine:
    #    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        #print("TRT volume size",size)
        #print("Engine binding shape",engine.get_binding_shape(binding))
        #print("Engine max batch",engine.max_batch_size)
    #    host_mem = cuda.pagelocked_empty(size, np.float32)
    #    cuda_mem = cuda.mem_alloc(host_mem.nbytes)
    
    #    bindings.append(int(cuda_mem))
    #    if engine.binding_is_input(binding):
    #        host_inputs.append(host_mem)
    #        cuda_inputs.append(cuda_mem)
    #    else:
    #        host_outputs.append(host_mem)
    #        cuda_outputs.append(cuda_mem)
        #print(host_inputs)
        #print(cuda_inputs)
    context = engine.create_execution_context()
    buffers = allocate_buffers(engine, 1)
    
    #context = engine.create_execution_context()
    ### context.set_binding_shape(1, (3, 608, 608))
    ### context.active_optimization_profile = 0
    if inference_with_CITY:
        for i, (images, labels, fname) in enumerate(valloader):
            #_, ext = os.path.splitext(os.path.basename((img_file)))
        
            #if ext not in [".png", ".jpg"]:
            #    continue
        
            ### Read input images in DIR
            #img_path = os.path.join(input_dir_path, img_file)
            #img_raw = read_img(img_path, proc_size)
            #img = img_raw.astype(np.float32)
            inputs, outputs, bindings, stream = buffers
            np.copyto(inputs[0].host, images.view(1,-1))
            
            #np.copyto(host_inputs[0], images.view(1,-1))

            start_time = time.perf_counter()
            #cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
            #context.execute(1, bindings=bindings)
            #cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
            #stream.synchronize()

            trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            elapsed_time = time.perf_counter() - start_time 
            
            ### Post-processing
            start_time = time.perf_counter()
        
            #pred_id = trt_outputs[0] = trt_outputs[0].reshape(1, n_classes, proc_size[0], proc_size[1])
            pred_id_val = trt_outputs[0].reshape(1, proc_size[0], proc_size[1])
            ### Reshape output image.
            #pred_id = host_outputs[0].reshape(1,n_classes,proc_size[0],proc_size[1])
            #if save_img:
            #    save_images2(images, pred_id, fname=fname[0], dir_path_id="../pred/", dir_path_img="../pred/", engine_img=True)
            #if save_img:
            #    filename = os.path.splitext(os.path.basename(single_test_img_path))[0]
            #    save_images(img, pred_id, fname=filename, dir_path_id="../predOrg/", dir_path_img="../predOrg/", engine_img=True) 

            if save_img:
                #filename = os.path.splitext(os.path.basename(single_test_img_path))[0]
                save_images2(images, pred_id_val, fname=fname[0], dir_path_id="../pred/", dir_path_img="../pred/", engine_img=True)  
            
            elapsed_time_post = time.perf_counter() - start_time 

            total_time += elapsed_time
            print("Inference time (iter {0:5d})--> Inference time:{1:3.5f}({2:3.5f}fps), Postprocess time:{3:3.5f}({4:3.5f}fps)".format(
                    i + 1, elapsed_time, 1/elapsed_time, elapsed_time_post, 1/elapsed_time_post
                )
            )
            i = i + 1
            
            gt = labels.numpy()
            #outputs = torch.tensor(pred_id, device = "cpu").float()
            #pred = outputs.data.max(1)[1].cpu().numpy()

            running_metrics.update(gt, pred_id_val)
            score, class_iou = running_metrics.get_scores()
    
        print("Total Frame Rate = %.2f fps" %(500/total_time ))
        
        for k, v in score.items():
            print(k, v)

        for i in range(n_classes):
            print(i, class_iou[i])
    elif inference_with_DIR:
        for img_file in os.listdir(input_dir_path):
            _, ext = os.path.splitext(os.path.basename((img_file)))
        
            if ext not in [".png", ".jpg"]:
                continue
        
            ### Read input images in DIR
            img_path = os.path.join(input_dir_path, img_file)
            img_raw = read_img(img_path, proc_size)
            img = img_raw.astype(np.float32)
            inputs, outputs, bindings, stream = buffers
            
            np.copyto(inputs[0].host, img.ravel())

            start_time = time.perf_counter()

            trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            elapsed_time = time.perf_counter() - start_time 
            
            ### Post-processing
            start_time = time.perf_counter()
        
            #pred_id = trt_outputs[0] = trt_outputs[0].reshape(1, n_classes, proc_size[0], proc_size[1])
            pred_id_val = trt_outputs[0].reshape(1, proc_size[0], proc_size[1])
            ### Reshape output image.
            #pred_id = host_outputs[0].reshape(1,n_classes,proc_size[0],proc_size[1])
            #if save_img:
            #    save_images2(images, pred_id, fname=fname[0], dir_path_id="../pred/", dir_path_img="../pred/", engine_img=True)
            #if save_img:
            #    filename = os.path.splitext(os.path.basename(single_test_img_path))[0]
            #    save_images(img, pred_id, fname=filename, dir_path_id="../predOrg/", dir_path_img="../predOrg/", engine_img=True) 

            if save_img:
                filename = os.path.splitext(os.path.basename(img_file))[0]
                save_images2(img, pred_id_val, fname=filename, dir_path_id="../pred/", dir_path_img="../out_rgb/", engine_img=True)  
            
            elapsed_time_post = time.perf_counter() - start_time 

            total_time += elapsed_time
            print("Inference time (iter {0:5d})--> Inference time:{1:3.5f}({2:3.5f}fps), Postprocess time:{3:3.5f}({4:3.5f}fps)".format(
                    i + 1, elapsed_time, 1/elapsed_time, elapsed_time_post, 1/elapsed_time_post
                )
            )
            i = i + 1
    
        print("Total Frame Rate = %.2f fps" %(i/total_time ))
    else:
        ### CUDA init
        # cuda.init()
        # device = cuda.Device(0)  # enter your Gpu id here
        # ctx = device.make_context()

        stream = cuda.Stream()



        ### Read test Image
        img = read_img(single_test_img_path, proc_size)

        inputs, outputs, bindings, stream = buffers
        #inputs[0].host = img_in

        #np.copyto(host_inputs[0], img.ravel())    
        np.copyto(inputs[0].host, img.ravel())

        

        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        print('Len of outputs: ', len(trt_outputs))

        #pred_id = trt_outputs[0] = trt_outputs[0].reshape(1, n_classes, proc_size[0], proc_size[1])
        #pred_id_val = trt_outputs[1] = trt_outputs[1].reshape(1, proc_size[0], proc_size[1])
        pred_id_val = trt_outputs[0] = trt_outputs[0].reshape(1, proc_size[0], proc_size[1])

        #cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0])   
        #context.execute_async(bindings=bindings, stream_handle=stream.handle)
        
        #cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0])
        #cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1])
        
        ### Transfer predictions back from the GPU.
        #[cuda.memcpy_dtoh_async(host_outputs, cuda_outputs, stream) for ostOut in cuda_outputs]   

        print("////////////////////")
        #print(host_outputs[1])

        #pred_id = host_outputs[0].reshape(1, n_classes, proc_size[0], proc_size[1])
        #pred_id_val = host_outputs[1].reshape(1, proc_size[0], proc_size[1])

        print("===============================")
        print(pred_id_val.shape)
        print(pred_id_val)
        #print(pred_id)
        #print(pred_id)
        ### Allocate Engine Memory
        #in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
        
        ### Copy image to cuda mememory from CPU memory
        #cuda.memcpy_htod(in_gpu, img.reshape(-1))
        
        ### Inference with context.execute
        #context.execute(1, [int(in_gpu), int(out_gpu)])
        
        ### Copy predicted image to cpu memory from CUDA memory
        #cuda.memcpy_dtoh(out_cpu, out_gpu)
        
        ### Reshape output image.
        #pred_id = out_cpu.reshape(1,n_classes,proc_size[0],proc_size[1])

        #if save_img:
        #    filename = os.path.splitext(os.path.basename(single_test_img_path))[0]
        #    save_images(img, pred_id, fname=filename, dir_path_id="../predOrg/", dir_path_img="../predOrg/", engine_img=True) 

        if save_img:
            filename = os.path.splitext(os.path.basename(single_test_img_path))[0]
            save_images2(img, pred_id_val, fname=filename, dir_path_id="../pred/", dir_path_img="../pred/", engine_img=True)  

        ### Context delete
        #ctx.pop()  # very important
        #del ctx