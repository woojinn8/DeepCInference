# Introduction
C++ Inference example for Deep Learning Framework
 - This project is a sample for reference when implementing the deep learning framework c++ code
 - It has been implemented for 8 types of deep learning frameworks
 - All samples are examples of image classification

---

# Supported

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
	- VULKAN is supported.
 - MXNet
 - MNN
	- OpenCL, Vulkan are supported.
 - Opencv
 	- You may need a version newer than 3.1 to use DNN module
	- OpenCV DNN library is used to load models of older frameworks
		- Caffe, TF 1.x, Darknet, Torch, DLDT and ONNX are supported

## Tested status matrix

|Framework|Linux|Android|Windows|
|:---:|:---:|:---:|:---:|
|Tensorflow|:heavy_check_mark:|:x:|:heavy_check_mark:|
|TFLite|:heavy_check_mark:|:heavy_check_mark:|:x:|
|Pytorch|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|ONNX|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|NCNN|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|MXNet|:white_check_mark:|:x:|:white_check_mark:|
|MNN|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|OpenCV DNN|:white_check_mark:|:x:|:white_check_mark:|

---

# Quick Start

## Requirement
- CMake v3.16.3
- OpenCV 3.4.16
- NDK r21e (for Android)

## Download prebuild library which you want to use

|Framework|Version|Download Link|
|:---:|:---:|:---:|
|Tensorflow|2.11.0|[download](https://www.tensorflow.org/install/lang_c)|
|TFLite|2.11.0|[download](https://drive.google.com/file/d/1UWtcjPMYYgcWtPht_2PDX3q4H2Gr8D0J/view?usp=share_link)|
|Pytorch|1.13.1|[download](https://pytorch.org/get-started/locally/)|
|ONNX|1.12.0|[download](https://github.com/microsoft/onnxruntime/releases)|
|NCNN|tag/20221128|[download](https://github.com/Tencent/ncnn/releases/tag/20221128)|
|MXNet|1.9.1|[download]|
|MNN|2.2.0|[download]|


## Build sample code using shell scripts.
  - Linux build
    - Set 3rdparty directory
	- Edit build.sh to set the framework you want
	- `sh build_linux.sh`
  - Android build
	- Edit build.sh to set the framework you want
	- `sh build_android.sh`
  - Windows build
	- `build`
