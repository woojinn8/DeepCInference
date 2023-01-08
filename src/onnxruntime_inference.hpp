#ifndef __ONNXRUNTIME_INFERENCE_HPP__
#define __ONNXRUNTIME_INFERENCE_HPP__

#include "DeepCInference.hpp"

// Header for ONNX
#include <experimental_onnxruntime_cxx_api.h>

// C++ API
class ONNXRuntimeCppEngine : public DeepCInference_Vision
{
public:
    ONNXRuntimeCppEngine() {};
	~ONNXRuntimeCppEngine() {
        session = nullptr;
    };

    int initialize(ConfigInfo setting);
    int inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor> &outputs);

private:
    std::shared_ptr<Ort::Experimental::Session> session;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
};
#endif