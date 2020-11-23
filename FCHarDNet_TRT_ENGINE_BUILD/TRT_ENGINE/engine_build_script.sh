#python3 torch_to_onnx.py --weight=./weights/hardnet_cityscapes_best_model_NEW_ORG.pkl --onnx=./FCHardNet_static608.onnx --width=608 --height=608 --config=./configs/hardnet.yml
python3 torch_to_onnx.py --weight=./weights/hardnet_cityscapes_best_model_0905_yolo_normalize.pkl --onnx=./FCHardNet_static608_NEW.onnx --width=608 --height=608 --config=./configs/hardnet.yml

# Static shape
#/TensorRT-7.1.3.4/bin/trtexec --explicitBatch \
./trtexec --explicitBatch \
          --onnx=FCHardNet_static608_NEW.onnx \
          --verbose=true \
          --workspace=4096 \
          --fp16 \
          --saveEngine=./FCHardNet_static608_fp16_NEW.engine

cp ./FCHardNet_static608_fp16_NEW.engine ../bin/