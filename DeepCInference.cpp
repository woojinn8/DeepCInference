#include "tensorflow_inference.hpp"
#include "tflite_inference.hpp"
#include "torch_inference.hpp"
#include "ncnn_inference.hpp"
#include "onnxruntime_inference.hpp"
// #include "mxnet_inference.hpp"
#include "mnn_inference.hpp"
#include "opencv_dnn_inference.hpp"

#include<algorithm>

void SubVectorByScalar(std::vector<float> &v, float k, int bias, int len){
    transform(v.begin() + bias, v.begin() + bias + len, v.begin() + bias, [k](float &c){ return c-k; });
}

void MulVectorByScalar(std::vector<float> &v, float k, int bias, int len){
    transform(v.begin() + bias, v.begin() + bias + len, v.begin() + bias, [k](float &c){ return c*k; });
}

void DivVectorByScalar(std::vector<float> &v, float k, int bias, int len){
    transform(v.begin() + bias, v.begin() + bias + len, v.begin() + bias, [k](float &c){ return c/k; });
}

std::shared_ptr<DeepCInference_Vision> DeepCInference_Vision::create(FrameworkType type)
{
    switch (type)
    {   
        // Tensorflow
        case FrameworkType::type_tensorflow : return std::make_shared<TensorflowEngine>(); break;

        // TFLite
        case FrameworkType::type_tflite_c : return std::make_shared<TfliteCEngine>(); break;  
        case FrameworkType::type_tflite_cpp : return std::make_shared<TfliteCppEngine>(); break;  
        
        // Pytoch
        case FrameworkType::type_torch : return std::make_shared<TorchEngine>(); break;
        
        // NCNN
        case FrameworkType::type_ncnn :return std::make_shared<NCNNEngine>(); break;
        
        // ONNX
        case FrameworkType::type_onnx : return std::make_shared<ONNXRuntimeCppEngine>(); break;

        // MXNet : ToDo
        case FrameworkType::type_mxnet : return nullptr; break;

        // MNN
        case FrameworkType::type_mnn :return std::make_shared<MNNEngine>(); break;

        // OpenCV DNN
        case FrameworkType::type_opencv_dnn : return std::make_shared<OpenCVdnnEngine>(); break;

        default : return nullptr; break;
    };
}

int DeepCInference_Vision::getSetting(std::string config_file_path)
{

    return Success;
}

int DeepCInference_Vision::createTensor(cv::Mat img, DeepCTensor &tensor, std::string name, bool isBGR)
{
    if (!img.empty())
    {
        tensor.img = img.clone();

        if (isBGR)
            cv::cvtColor(tensor.img, tensor.img, cv::COLOR_BGR2RGB);

        // set shape
        tensor.shape.push_back(1);
        tensor.shape.push_back(img.channels());
        tensor.shape.push_back(img.rows);
        tensor.shape.push_back(img.cols);

        tensor.data.resize(tensor.shape[1] * tensor.shape[2] * tensor.shape[3]);
        
        // preprocess
        cv::Mat fimg;
        std::vector<cv::Mat> inputImage;

        tensor.storedType = StoredType::BCHW;
        img.convertTo(fimg, CV_32FC3);
        split(fimg, inputImage);

        int size_img = img.rows * img.cols;
        for (int i = 0; i < 3; i++)
            std::memcpy(tensor.data.data() + i * size_img, inputImage[i].data, sizeof(float) * size_img);
    }
    tensor.name = name;   

    return Success;
}

int DeepCInference_Vision::normalize(DeepCTensor& tensor)
{
    if (tensor.data.size() == 0)
    {
        std::cout << "No data" << std::endl;
        return Fail_Tensor_Preprocess;
    }

    if(tensor.storedType == StoredType::BHWC)
    {
        cv::Mat fimg;
        std::vector<cv::Mat> inputImage;

        tensor.storedType = StoredType::BCHW;
        cv::cvtColor(tensor.img, tensor.img, cv::COLOR_BGR2RGB);
        tensor.img.convertTo(fimg, CV_32FC3);
        split(fimg, inputImage);

        int size_img = tensor.img.rows * tensor.img.cols;
        for (int i = 0; i < 3; i++)
            std::memcpy(tensor.data.data() + i * size_img, inputImage[i].data, sizeof(float) * size_img);
    }

    // preprocess
    int width = tensor.shape[3];
    int height = tensor.shape[2];
    int size = width*height;
    for (int c = 0; c < tensor.shape[1]; c++)
    {
        SubVectorByScalar(tensor.data, this->norm_mean[c], c*size, size);
        DivVectorByScalar(tensor.data, this->norm_std[c], c*size, size);
    }
    return Success;
}

int DeepCInference_Vision::getTensorSize(const DeepCTensor &tensor)
{
    int size = 1;
    for (int i = 0; i < tensor.shape.size(); i++)
        size *= tensor.shape[i];
    return size;
}

int DeepCInference_Vision::copyDataToTensor(float* output, DeepCTensor &tensor)
{
    if(output == nullptr)
        return Fail_Tensor_Copy;

    tensor.data.resize(getTensorSize(tensor));
    std::memcpy(tensor.data.data(), output, sizeof(float) * getTensorSize(tensor));
    return Success;
}