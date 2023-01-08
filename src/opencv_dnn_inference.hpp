#ifndef __OPENCV_DNN_INFERENCE_HPP__
#define __OPENCV_DNN_INFERENCE_HPP__


#include "DeepCInference.hpp"

// Header for OPENCV_DNN
#include "opencv2/opencv.hpp"
 

class OpenCVdnnEngine : public DeepCInference_Vision
{

public:
    OpenCVdnnEngine() {};
	~OpenCVdnnEngine() {
    };

    int initialize(ConfigInfo setting);
    int inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor> &outputs);

private:
   float threshold;
   cv::dnn::Net net;

};
#endif