#ifndef __DEEPCINFERENCE_HPP__
#define __DEEPCINFERENCE_HPP__

#include <iostream>
#include <chrono>
#include <memory>
#include "opencv2/opencv.hpp"
#include <unistd.h>

enum errorMessage
{
    // Sucess, Fail
    Success,
    Fail,

    // Tensor
    Fail_Tensor_Create,
    Fail_Tensor_Copy,
    Fail_Tensor_Preprocess,
    Fail_Tensor_Convert_CHW_HWC,

    // initialize
    Fail_Init,
    Fail_Init_Load_Model,
    Fail_Init_Set_Thread,
    Fail_Init_Set_GPU,
    Fail_Init_Set_FP16,
    Fail_Init_Set_Input_Output,

    // Inference
    Fail_Infer,
    Fail_Infer_Set_Inputs,
    Fail_Infer_Invoke,
    Fail_Infer_Get_Outputs,
};

// Config information to inference Deeplearning model
typedef struct _ConfigInfo
{   
    // Basically, use variable named model_file for model file path
    // If framework need config or parameter file, use param_file
    std::string model_file; 
    std::string param_file; 

    int num_thread;
    bool use_gpu;
    bool use_fp16;

    // Variable for big.Little solution
    // 0 : All Core
    // 1 : Little Core
    // 2 : Big Core
    int powersave;

    // Name list of inputs/outputs
    // To use ncnn, at least one input/output node name should be inserted
    std::vector<std::string> input_node_name;
    std::vector<std::string> output_node_name;

    float mean[3];
    float std[3];

    _ConfigInfo()
    {
        model_file = "";
        param_file = "";
        num_thread = 1;
        use_gpu = false;
        use_fp16 = false;
        powersave = 0;
    }

} ConfigInfo;

enum FrameworkType
{
    type_tensorflow,
    type_tflite_c,
    type_tflite_cpp,
    type_torch,
    type_ncnn,
    type_onnx,
    type_mxnet,
    type_mnn,    
    type_opencv_dnn
};

enum StoredType
{
    BHWC, // Stored in the order of [Batch - Height - Width - Channel] - default
    BCHW, // Stored in the order of [Batch - Channel - Height - Width]
    None  // Non type (for output)
};

// Type for set/get value about input/output
typedef struct _DeepCTensor
{

    cv::Mat img;

    // input image data
    std::vector<float> data;

    // data stored type
    StoredType storedType = StoredType::BHWC;

    // shape of tensor
    std::vector<int> shape;

    // node name of tensor(to use ncnn, node name should be inserted)
    std::string name;

} DeepCTensor;


class DeepCInference_Vision
{
public:
    static std::shared_ptr<DeepCInference_Vision> create(FrameworkType type);

    DeepCInference_Vision(){};

    virtual ~DeepCInference_Vision(){};

    virtual int initialize(ConfigInfo setting) = 0;
    virtual int inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor> &outputs) = 0;

    int getSetting(std::string config_file_path);
    int createTensor(cv::Mat img, DeepCTensor &tensor, std::string name = "", bool isBGR = true);
    int normalize(DeepCTensor& tensor);
    int getTensorSize(const DeepCTensor &tensor);
    int copyDataToTensor(float *output, DeepCTensor &tensor);

    float norm_mean[3];
    float norm_std[3];
};

void SubVectorByScalar(std::vector<float> &v, float k, int bias, int len);
void MulVectorByScalar(std::vector<float> &v, float k, int bias, int len);
void DivVectorByScalar(std::vector<float> &v, float k, int bias, int len);

#endif