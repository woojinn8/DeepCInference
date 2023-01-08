# Introduction
C++ Inference example for Deep Learning Framework
 - This project is a sample for reference when implementing the deep learning framework c++ code
 - It has been implemented for 8 types of deep learning frameworks
 - All samples are examples of image classification

# 1. Requirement
- CMake v3.16.3
- OpenCV 3.4.16
- NDK r21e (for Android)


# 2.Quick Start
1. Download prebuild library which you want to use

|Framework|Version|Download Link|
|:---:|:---:|:---:|
|Tensorflow|2.11.0|download|
|TFLite|2.11.0|download|
|Pytorch|1.13.1|download|
|ONNX|1.10.0|download|
|NCNN|tag/20221128|download|
|MXNet|1.9.1|download|
|MNN|-|download|
|OpenCV|3.4.16|download|


2. Build inference code using shell scripts.
  - Linux build
    - Set 3rdparty directory
	- Edit build.sh to set the framework you want
	- `sh build_linux.sh`
  - Android build
	- Edit build.sh to set the framework you want
	- `sh build_android.sh`
  - Windows build
	- `build`

# 3.Supported

## Target Platform
 - Linux(x64)
 - Android(aarch64)
 - Windows(x64)

## Target Framework
 - Tensorflow
 - TFLite
	- CPU, GPU, XNNPACK and NNAPI are supported
	- Edge TPU will updated soon
 - Pytorch
 - ONNX
 - NCNN
 - MXNet
 - MNN
 - Opencv
 	- You may need a version newer than 3.1 to use DNN module
	- OpenCV DNN library is used to load models of older frameworks
		- Caffe, TF 1.x, Darknet, Torch, DLDT and ONNX are supported

## Status

|Framework|Linux|Android|Windows|
|:---:|:---:|:---:|:---:|
|Tensorflow|:heavy_check_mark:|-|:heavy_check_mark:|
|TFLite|:heavy_check_mark:|:heavy_check_mark:|-|
|Pytorch|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|ONNX|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|NCNN|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|MXNet|:white_check_mark:|-|:white_check_mark:|
|MNN|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|OpenCV|:white_check_mark:|-|:white_check_mark:|

