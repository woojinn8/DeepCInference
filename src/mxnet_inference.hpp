#ifndef __MXNET_INFERENCE_HPP__
#define __MXNET_INFERENCE_HPP__


#include "DeepCInference.hpp"

// Header for MXNET
#include "mxnet/c_api.h"
#include "mxnet/tuple.h"
#include "mxnet/c_predict_api.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "mxnet-cpp/initializer.h"

using namespace std;
using namespace mxnet::cpp;

using namespace experimental::filesystem;

class MxnetEngine : public DeepCInference_Vision
{

public:
    MxnetEngine() {};
	~MxnetEngine() {
        
    };

    int initialize(ConfigInfo setting);
    int inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor> &outputs);

private:
  
    mxnet::cpp::Context m_CTX = Context::cpu();

	map<string, NDArray> m_ArgsMap, m_AuxMap;
	mxnet::cpp::Symbol m_Symbols;

	mxnet::cpp::Executor* m_Executor;
	mxnet::cpp::Shape m_InputShape, m_OutputShape;
	std::vector<float> m_ImageData;
	cv::Mat m_FloatImage;
	

};
#endif