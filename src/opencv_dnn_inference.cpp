#include "opencv_dnn_inference.hpp"

#include "mnn_inference.hpp"

int OpenCVdnnEngine::initialize(ConfigInfo setting)
{
	// Find framework and Load model
	std::string framework = "";
	const std::string modelExt = setting.model_file.substr(setting.model_file.rfind('.') + 1);
    const std::string paramExt = setting.param_file.substr(setting.param_file.rfind('.') + 1);
    if (modelExt == "caffemodel" || paramExt == "caffemodel" || modelExt == "prototxt" || paramExt == "prototxt")
    {
        if (modelExt == "prototxt" || paramExt == "caffemodel")
            std::swap(setting.model_file, setting.param_file);
        framework == "caffe";
        this->net = cv::dnn::readNetFromCaffe(setting.model_file, setting.param_file);
    }
    else if (modelExt == "pb" || paramExt == "pb" || modelExt == "pbtxt" || paramExt == "pbtxt")
    {
        if (modelExt == "pbtxt" || paramExt == "pb")
            std::swap(setting.model_file, setting.param_file);
        framework == "tensorflow"; 
        this->net = cv::dnn::readNetFromTensorflow(setting.model_file, setting.param_file);
    }
    else if (modelExt == "t7" || modelExt == "net" || paramExt == "t7" || paramExt == "net")
    {
        framework == "torch";
        this->net = cv::dnn::readNetFromTorch(setting.model_file);
    }
    else if (modelExt == "weights" || paramExt == "weights" || modelExt == "cfg" || paramExt == "cfg")
    {
        if (modelExt == "cfg" || paramExt == "weights")
            std::swap(setting.model_file, setting.param_file);
        framework == "darknet";
        this->net = cv::dnn::readNetFromDarknet(setting.model_file, setting.param_file);
    }
    else if (modelExt == "bin" || paramExt == "bin" || modelExt == "xml" || paramExt == "xml")
    {
        if (modelExt == "xml" || paramExt == "bin")
            std::swap(setting.model_file, setting.param_file);
        framework == "dldt";
        this->net = cv::dnn::readNetFromModelOptimizer(setting.model_file, setting.param_file);
    }
    else if (modelExt == "onnx")
    {
        framework == "onnx";
        this->net = cv::dnn::readNetFromONNX(setting.model_file);
    }
    else{
        this->net = cv::dnn::readNet(setting.model_file, setting.param_file, framework);
    }

    if (this->net.empty()) 
		return Fail_Init_Load_Model;
	
     // Set Preferable Backend
    this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);

    // Set Preferable Target
    //this->net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);

	std::memcpy(this->norm_mean, setting.mean, sizeof(float) * 3);
	std::memcpy(this->norm_std, setting.std, sizeof(float) * 3);

	return Success;
}


int OpenCVdnnEngine::inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor> &outputs)
{   

	if (inputs.size() == 0)
	{
		return Fail_Infer_Set_Inputs;
	}

    for (int i = 0; i < inputs.size(); i++)
    {        
        int result_norm = this->normalize(inputs[i]);
        if (result_norm != Success)
        {
            return Fail_Infer_Set_Inputs;
        }

        cv::Mat inputBlob = cv::dnn::blobFromImage(inputs[i].img);
        net.setInput(inputBlob, inputs[i].name);
    }
    
    // get outputs
    for (int i=0; i<outputs.size(); i++)
    {
        cv::Mat res = net.forward(outputs[i].name);
        
        int dims = res.dims;
        int size_vec = 1;
        outputs[i].shape.clear();
        for(int j=0; j < dims; j++)
        {
            outputs[i].shape.push_back(res.size[j]);
            size_vec *= res.size[j];
        }

        std::vector<float> res_float(size_vec);
        std::memcpy(res_float.data(), ptrScore, sizeof(float) * size_vec);

        copyDataToTensor(res_float.data(), outputs[i]);
    }

    return Success;
}
