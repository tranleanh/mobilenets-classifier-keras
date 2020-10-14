# Train/Test a Classifier with MobileNet/MobileNetv2

Object Classification with MobileNet/MobileNetv2 on Custom Datasets.

## Dependencies
- Python 3.6+
- OpenCV
- Tenforflow 2.x
- Keras

## Preparation
- (Optional) Verify your GPU(s) in [verify_gpu.py](https://github.com/tranleanh/mobilenets-classifier-keras/blob/main/verify_gpu.py) 
- (Optional) Config Classification Head (Top Layers/Activation Functions) in [mobilenet.py](https://github.com/tranleanh/mobilenets-classifier-keras/blob/main/mobilenet.py) 
or [mobilenetv2.py](https://github.com/tranleanh/mobilenets-classifier-keras/blob/main/mobilenetv2.py)
- Training: Config Dataset and Number of Categories in [train.py](https://github.com/tranleanh/mobilenets-classifier-keras/blob/main/train.py)
- Testing: Config Test Images in [test.py](https://github.com/tranleanh/mobilenets-classifier-keras/blob/main/test.py)

## Train/Test
### 1. Train
```bashrc
$ python train.py
```

### 2. Test
```bashrc
$ python test.py
```

Oct. 2020

Tran Le Anh
