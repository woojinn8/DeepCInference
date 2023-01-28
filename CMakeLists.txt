cmake_minimum_required(VERSION 3.10)
project(test_DeepCInference C CXX)

OPTION (BUILD_ANDROID "Build for Android" OFF)

# Basic Path Setting
set(ATTRIBUTE PRIVATE)
set(THIRD_PARTY ${CMAKE_CURRENT_LIST_DIR}/0_3rdparty)

if(USE_ANDROID)
set(BUILD_ANDROID ${CMAKE_CURRENT_LIST_DIR}/build_android)
else()
set(BUILD_PATH ${CMAKE_CURRENT_LIST_DIR}/build_linux)
endif()


# TF/TFLITE Path Setting
set(TF_INC 
${THIRD_PARTY}/tensorflow_tflite_2.10.0
${THIRD_PARTY}/tensorflow_tflite_2.10.0/flatbuffers/include
${THIRD_PARTY}/tensorflow_tflite_2.10.0/abseil-cpp
)

if(BUILD_ANDROID)
  set(TF_LIB 
    ${THIRD_PARTY}/tensorflow_tflite_2.10.0/lib_linux/libtensorflowlite.so
    ${THIRD_PARTY}/tensorflow_tflite_2.10.0/lib_linux/libtensorflowlite_c.so
  )
  set(TFLITE_GPU_LIB ${THIRD_PARTY}/tensorflow_tflite_2.10.0/lib_linux/libtensorflowlite_gpu_delegate.so)
elseif(UNIX)
  set(TF_LIB 
    ${THIRD_PARTY}/tensorflow_tflite_2.10.0/lib_android/libtensorflowlite.so
    ${THIRD_PARTY}/tensorflow_tflite_2.10.0/lib_android/libtensorflowlite_c.so
  )
elseif(WIN32)
set(TF_LIB 
${THIRD_PARTY}/tensorflow_tflite_2.10.0/lib_android/libtensorflowlite.lib
${THIRD_PARTY}/tensorflow_tflite_2.10.0/lib_android/libtensorflowlite_c.lib
)
endif()

# Pytorch Path Setting
set(TORCH_INC ${THIRD_PARTY}/torch_1.13/include)
set(TORCH_LIB ${THIRD_PARTY}/torch_1.13/lib_linux)


# NCNN Path Setting
set(NCNN_INC ${THIRD_PARTY}/ncnn_tag_20220729/include/ncnn)
set(NCNN_LIB ${THIRD_PARTY}/ncnn_tag_20220729/lib_linux/libncnn.so)

# ONNX Path Setting
set(ONNXRUNTIME_INC ${THIRD_PARTY}/onnxruntime_1.12.0/include/onnxruntime/core/session)
set(ONNXRUNTIME_LIB ${THIRD_PARTY}/onnxruntime_1.12.0/lib_linux/libonnxruntime.so)


# MXNet Path Setting


# MNN Path Setting
set(MNN_INC ${THIRD_PARTY}/mnn_2.2.0/include)
set(MNN_LIB ${THIRD_PARTY}/mnn_2.2.0/lib_linux/libMNN.so)
#set(MNN_VULKAN_LIB ${THIRD_PARTY}/mnn_2.2.0/libs/linux_x64/libMNN_Vulkan.so)

# OpenCV Path Setting
set(OPENCV_INC ${THIRD_PARTY}/opencv_3.4.16/include)
set(OPENCV_LIB ${THIRD_PARTY}/opencv_3.4.16/lib_linux)


INCLUDE_DIRECTORIES (
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/src
)

add_executable(test_DeepCInference 
  ${CMAKE_CURRENT_LIST_DIR}/test_DeepCInference.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DeepCInference.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow_inference.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tflite_inference.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/torch_inference.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/ncnn_inference.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/onnxruntime_inference.cpp
  #${CMAKE_CURRENT_LIST_DIR}/src/mxnet_inference.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/mnn_inference.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/opencv_dnn_inference.cpp
)

# Link TF
target_include_directories(test_DeepCInference PRIVATE ${TF_INC})
target_link_libraries(test_DeepCInference ${TF_LIB})
target_link_libraries(test_DeepCInference ${TF_GPU_LIB})

# Link Torch
target_include_directories(test_DeepCInference ${ATTRIBUTE} ${TORCH_INC})
target_link_libraries(test_DeepCInference 
  ${TORCH_LIB}/libc10.so 
  ${TORCH_LIB}/libc10_cuda.so
  ${TORCH_LIB}/libtorch_cpu.so
  ${TORCH_LIB}/libtorch_cuda.so
  ${TORCH_LIB}/libtorch.so
)

# Link MNN
target_include_directories(test_DeepCInference ${ATTRIBUTE} ${MNN_INC})
target_link_libraries(test_DeepCInference ${MNN_LIB} ${MNN_VULKAN_LIB})

# Link NCNN
target_include_directories(test_DeepCInference ${ATTRIBUTE} ${NCNN_INC})
target_link_libraries(test_DeepCInference ${NCNN_LIB})

# Link ONNX
target_include_directories(test_DeepCInference ${ATTRIBUTE} ${ONNXRUNTIME_INC})
target_link_libraries(test_DeepCInference ${ONNXRUNTIME_LIB})

# Link OpenCV
target_include_directories(test_DeepCInference ${ATTRIBUTE} ${OPENCV_INC})
target_link_libraries(test_DeepCInference
  ${OPENCV_LIB}/libopencv_calib3d.so
  ${OPENCV_LIB}/libopencv_core.so
  ${OPENCV_LIB}/libopencv_dnn.so
  ${OPENCV_LIB}/libopencv_features2d.so
  ${OPENCV_LIB}/libopencv_highgui.so
  ${OPENCV_LIB}/libopencv_imgcodecs.so
  ${OPENCV_LIB}/libopencv_imgproc.so
  ${OPENCV_LIB}/libopencv_ml.so
  ${OPENCV_LIB}/libopencv_photo.so
  ${OPENCV_LIB}/libopencv_video.so  
)
  
# Link Others
find_package(OpenCL)
if(OpenCL_Found)
  target_include_directories(test_DeepCInference ${ATTRIBUTE} ${OpenCL_INCLUDE_DIRS})
  target_link_libraries(test_DeepCInference ${OpenCL_LIBRARIES})
endif()

target_link_libraries(test_DeepCInference
  EGL 
  GLESv2
)

if(USE_ANDROID)
find_library(ANDROID_LOG_LIB log)
target_link_libraries(test_DeepCInference log)
endif()
