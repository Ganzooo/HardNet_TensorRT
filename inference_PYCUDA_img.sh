WORKING_DIR=INFERENCE_PYCUDA
cd ./$WORKING_DIR 
#python3 ./inference_pycuda_CLEAN.py --classes=19 --showResult=True --inputTestType=CITY --input=../img/berlin_000000_000019_leftImg8bit.png --height=608 --width=608 --engine=../bin/FCHardNet_static608_fp16_val_NEW2.engine 
#python3 ./inference_pycuda_CLEAN.py --classes=19 --showResult=True --inputTestType=CITY --input=../img/berlin_000000_000019_leftImg8bit.png --height=608 --width=608 --engine=../bin/FCHardNet_static608_fp16_val.engine 
python3 ./inference_pycuda_dynamic.py --classes=19 --showResult=True --inputTestType=S --input=../img/berlin_000000_000019_leftImg8bit.png --height=608 --width=608 --engine=../bin/FCHardNet_dynamic_3_608_fp16.engine 