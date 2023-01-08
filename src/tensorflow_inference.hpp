#ifndef __TF_INFERENCE_HPP__
#define __TF_INFERENCE_HPP__


#include "DeepCInference.hpp"

// Header for TF
#include <tensorflow/c/c_api.h>
//#include <tensorflow/core/platform/env.h>
//#include <tensorflow/core/public/session.h>


class TensorflowEngine : public DeepCInference_Vision
{

public:
    TensorflowEngine() {};
	~TensorflowEngine() {
        TF_DeleteBuffer(run_options);
        TF_DeleteSessionOptions(session_options);
        TF_DeleteSession(session, status);
        TF_DeleteGraph(graph);
        TF_DeleteStatus(status);
    };

    int initialize(ConfigInfo setting);
    int inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor> &outputs);

private:
    TF_Buffer *run_options = TF_NewBufferFromString("", 0);
    TF_Session* session;
    TF_SessionOptions *session_options = TF_NewSessionOptions();
    TF_Graph *graph = TF_NewGraph();
    TF_Status *status = TF_NewStatus();
    std::array<char const *, 1> tags{"serve"};

};
#endif