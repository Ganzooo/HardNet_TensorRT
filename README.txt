***FCHardNet_TRT tested with following enviroment:
  - OS: Ubuntu18.04 / Ubuntu16.04
  - Python: v3.6
  - Torch: v1.6
  - TensorRT: TensorRT-7.1.3
      - TRT installation: https://github.com/NVIDIA/TensorRT
  - CUDA: 10.02

***Usage of FCHardNET_TRT:
1. Build TRT engine 
  - sh engine_build.sh
     * Process:
        - Read FCHardNet model from pytorch 
        - Generate ONNX file 
        - Generate TensorRT engine from ONNX file 
           - use "trtexec" file 
                - if its gives error --> replace it from /TensorRT-7.1.3/bin 
        - Generated engine COPY to /bin folder.
            - ./bin/FCHardNet_static608_fp16.engine

2. Inference PYCUDA
   - sh inference_PYCUDA_img.sh
     *Argument details in python:
        --engine=../bin/FCHardNet_static608_fp16.engine --> engine file name
        --width=608					--> target input width
    	--height=608					--> target input height
	--input=../img/berlin_000000_000019_leftImg8bit.png --> input file name or input directory depends on inputTestType
        --inputTestType= single				    --> input test TYPE: single / dir
    	--showResult=True				    --> Show segmentation result, result file saved in ./pred folder
        --classes=19                                        --> number of class

3. Inference C-API
   - inference_CAPI_img.sh
	--C=3 					--> input channel
        --H=608					--> input height
        --W=608                                 --> input width
        --class=19                              --> number of class
        --img=on                                --> image test
        --input=./img/berlin_000000_000019_leftImg8bit.png  --> input test image
        --engine=./FCHardNet_static608_fp16.engine          --> input TensorRT engine
        --show_result=on                                    --> Show segmentation result, result file saved in ./pred folder

4. For inference C-API require ==> ./runFCHardNet file
   - it can be build with ./build_FCHardNet.sh in /INFERENCE_CAPI
   
   
