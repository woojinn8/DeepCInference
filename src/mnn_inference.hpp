#ifndef __MNN_INFERENCE_HPP__
#define __MNN_INFERENCE_HPP__


#include "DeepCInference.hpp"

// Header for MNN
#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"

class MNNEngine : public DeepCInference_Vision
{

public:
    MNNEngine() {};
	~MNNEngine() {
        interpreter.reset();
        session = nullptr;
    };

    int initialize(ConfigInfo setting);
    int inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor> &outputs);

private:
   float threshold;

	std::shared_ptr<MNN::Interpreter> interpreter;
	MNN::Session *session;
	
	MNN::ScheduleConfig config_;
	MNN::BackendConfig backendConfig_;

    MNN::Tensor *inputTensor;
	MNN::Tensor *outputTensor;

};
#endif