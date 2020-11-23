
from itertools import izip
from std_msgs.msg  import  String 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ImageMsg

import sys
import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda 
import time
import os
import cv2
import torch
import rospy 
import threading

from sensor_msgs.msg import CompressedImage

from trt_wrapper_common import TRTInference, save_images, img_preprocessing

TRT_LOGGER = trt.Logger()

## Documentation for a function.
#
#  FCHarDNet Inference 
class FCHarDNet_Inference(object):
    """! The sensor base class.
    Defines the base class utilized by all sensors.
    """
    ## Initialize the TensorRT engine
    # @param self The object pointer. 
    def __init__(self):
	## Engine path
        self.ENGINE_PATH = "/workspace/NETWORK/ROS/FCHarDNet_TRT/fchardnet_ws/src/bin/FCHardNet_static608_fp16_val.engine"
	
	## Batch size of inference
        self.batch_size = 1
        
	## Number of classes
        self.num_classes = 19

	## Inference image size
        self.proc_size = [608, 608]

	## Initialize the TRT Inference model
        self.trt_wrapper = TRTInference(self.ENGINE_PATH, self.batch_size, self.num_classes, self.proc_size)

        ## ROS PART Initialize
        self.bridge_ROS = CvBridge()
        self.loop_rate = rospy.Rate(1)
        self.pub = rospy.Publisher('Image_Label', String, queue_size=1)
        
    ## Callback function of FCHarDNet inference
    # @param self: The object pointer
    def callback(self):
        ## Get image data from ROS Topic
	rospy.Subscriber("/eth_cam_1/camera/image_raw/compressed", CompressedImage, msg)
	
        rospy.loginfo('Image received...')
        
        ## Get ROS topic images to cv_image    	
        cv_image = self.bridge_ROS.compressed_imgmsg_to_cv2(msg, 'bgr8')

        ## Prepare image for TensorRT Engine
        img = self.trt_wrapper.img_preprocessing(cv_image, self.proc_size)

	## Inference 
        pred_output = self.trt_wrapper.infer(img)
        
	## Reshape the predicted output of TensorRT
        pred_id = pred_output[0].reshape(self.batch_size,self.num_classes,self.proc_size[0],self.proc_size[1])
        pred_id_val = pred_output[1].reshape(self.batch_size,self.proc_size[0],self.proc_size[1])

	## Save result image
        self.trt_wrapper.save_images(img, pred_id, fname=filename, dir_path_id="./FCHarDNet_TRT/fchardnet_ws/src/pred/", dir_path_img="./FCHarDNet_TRT/fchardnet_ws/src/pred/", engine_img=True)

def main():
    """! Main program entry."""
    rospy.init_node("listener", anonymous=True)
    infer = ModelData()
    infer.callback()

if __name__ == '__main__':
    main()

