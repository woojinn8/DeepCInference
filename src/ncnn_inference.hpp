#ifndef __NCNN_INFERENCE_HPP__
#define __NCNN_INFERENCE_HPP__

#include "DeepCInference.hpp"

// Header for ncnn
#include "gpu.h"
#include "net.h"
//#include "mat.h"
#include "cpu.h"
//#include "datareader.h"

#define VULKAN_BUILD

class NCNNEngine : public DeepCInference_Vision
{
public:

    NCNNEngine(){};
    ~NCNNEngine(){
        net.clear();
    };

    int initialize(ConfigInfo setting);
    int inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor>& outputs);

private:
    ncnn::Net net;

    std::string model_bin;
    std::string model_param;
    int num_thread;
    int use_gpu;
    int use_fp16;
    int power_save;
};

#endif