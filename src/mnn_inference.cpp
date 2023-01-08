#include "mnn_inference.hpp"

int MNNEngine::initialize(ConfigInfo setting)
{
	this->interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(setting.model_file.c_str()));
	if(this->interpreter == NULL)
		return Fail_Init_Load_Model;
	
	MNN::ScheduleConfig config;
    if(setting.use_gpu)
        config.type = MNN_FORWARD_VULKAN;
    else
        config.type = MNN_FORWARD_CPU;

	config.numThread = setting.num_thread;
	
    MNN::BackendConfig backendConfig;
    backendConfig.memory = MNN::BackendConfig::MemoryMode::Memory_High;
	backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_High;
    backendConfig.power = MNN::BackendConfig::PowerMode::Power_High;

	config.backendConfig = &backendConfig;

	this->session = interpreter->createSession(config);

	std::memcpy(this->norm_mean, setting.mean, sizeof(float) * 3);
	std::memcpy(this->norm_std, setting.std, sizeof(float) * 3);

	inputTensor = nullptr;
	outputTensor = nullptr;

	return Success;
}


int MNNEngine::inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor> &outputs)
{
	if (inputs.size() == 0)
	{
		return Fail_Infer_Set_Inputs;
	}

	for (int i = 0; i < inputs.size(); i++)
	{
		inputTensor = interpreter->getSessionInput(session, inputs[i].name.c_str());
		int w, h, c;
		if (inputs[i].storedType == StoredType::BHWC)
		{
			h = inputs[i].shape[1];
			w = inputs[i].shape[2];
			c = inputs[i].shape[3];
		}
		else
		{
			c = inputs[i].shape[1];
			h = inputs[i].shape[2];
			w = inputs[i].shape[3];
		}
		this->interpreter->resizeTensor(inputTensor, {1, c, h, w});
		this->interpreter->resizeSession(this->session);
		
		std::shared_ptr<MNN::CV::ImageProcess> pretreat(
			MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, this->norm_mean, 3, this->norm_std, 3));
		pretreat->convert(inputs[i].img.data, w, h, w*c, inputTensor);
		
	}
	
	this->interpreter->runSession(this->session);

	for(int i=0; i<outputs.size(); i++)
	{
		outputTensor = interpreter->getSessionOutput(this->session, outputs[i].name.c_str());
		MNN::Tensor output_host(outputTensor, outputTensor->getDimensionType());
		outputTensor->copyToHostTensor(&output_host);
		float *output_data = output_host.host<float>();

		outputs[i].shape = output_host.shape();

		copyDataToTensor(output_data, outputs[i]);
	}
	
	return Success;
}