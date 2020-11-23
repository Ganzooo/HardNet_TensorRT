#python3 torch_to_onnx.py --weight=./weights/hardnet_cityscapes_best_model_NEW_ORG.pkl --onnx=./FCHardNet_static608.onnx --width=608 --height=608 --config=./configs/hardnet.yml
python3 torch_to_onnx.py --dynamic=True --weight=./weights/hardnet_cityscapes_best_model_0905_yolo_normalize.pkl --onnx=./FCHardNet_dynamic608.onnx --width=608 --height=608 --config=./configs/hardnet.yml

# Static shape
#./trtexec --explicitBatch \
#          --onnx=FCHardNet_dynamic608.onnx \
#          --verbose=true \
#          --workspace=4096 \
#          --fp16 \
#          --minShapes=input:1x3x608x608 \
#          --optShapes=input:5x3x608x608 \
#          --maxShapes=input:8x3x608x608 \
#          --shapes=input:3x3x1024x2048 \
#          --saveEngine=./FCHardNet_dynamic608_fp16.engine

./trtexec --explicitBatch \
          --onnx=FCHardNet_dynamic608.onnx \
          --verbose=true \
          --workspace=4096 \
          --fp16 \
          --shapes=input:3x3x608x608 \
          --saveEngine=./FCHardNet_dynamic_0_608_fp16.engine

cp ./FCHardNet_dynamic_0_608_fp16.engine ../../bin/