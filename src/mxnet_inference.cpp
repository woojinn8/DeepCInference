#include "mxnet_inference.hpp"

int MxnetEngine::initialize(ConfigInfo setting)
{
    
    if (use_gpu)
		m_CTX = Context::gpu();

	if (!exists(setting.model_file)) { cout << "No " << setting.model_file << endl; return false; }
	m_Symbols = Symbol::Load(setting.model_file);

	if (!exists(setting.param_file)) { cout << "No " << setting.param_file << endl; return false; }
	map<string, NDArray> parameters;
	NDArray::Load(setting.param_file, 0, &parameters);

	if (!m_ArgsMap.empty()) { m_ArgsMap.clear(); }
	if (!m_AuxMap.empty()) { m_AuxMap.clear(); }
	for (const auto& pair : parameters)
	{
		string type = pair.first.substr(0, 4);
		string name = pair.first.substr(4);
		if (type == "arg:")
			m_ArgsMap[name] = pair.second.Copy(m_CTX);
		else if (type == "aux:")
			m_AuxMap[name] = pair.second.Copy(m_CTX);
	}
	NDArray::WaitAll();

	m_InputShape = Shape(1, 3, 128, 128);
	m_ArgsMap["data"] = NDArray(m_InputShape, m_CTX, false, 0);
	m_InputImage.clear();
	m_ImageData.resize(1 * 3 * IMAGE_SIZE);

	float* pImg = (float*)m_ImageData.data();

	for (int ch = 0; ch < 3; ch++, pImg += IMAGE_SIZE)
		m_InputImage.push_back(Mat(Size(128, 128), CV_32FC1, pImg));

	vector<NDArray> aArrays, gArrays, xArrays;
	vector<OpReqType> gReqs;
	m_Symbols.InferExecutorArrays(m_CTX, &aArrays, &gArrays, &gReqs, &xArrays,
		m_ArgsMap, map<string, NDArray>(), map<string, OpReqType>(), m_AuxMap);
	for (auto& i : gReqs)
		i = OpReqType::kNullOp;

	if (m_Executor) { delete m_Executor; m_Executor = NULL; }
	m_Executor = new Executor(m_Symbols, m_CTX, aArrays, gArrays, gReqs, xArrays);

	aArrays.clear();
	gArrays.clear();
	xArrays.clear();
	gReqs.clear();

	// warming up
	for (int i = 0; i < 3; i++)
		m_Executor->Forward(false);
	NDArray::WaitAll();

	m_OutputShape = m_Executor->outputs[0].GetShape();

	return true;

    std::memcpy(this->norm_mean, setting.mean, sizeof(float)*3);
    std::memcpy(this->norm_std, setting.std, sizeof(float)*3);

    return Success;
}


int MxnetEngine::inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor>& outputs)
{
    
	cv::Mat fImage;
    std::vector<cv::Mat> m_InputImage;
	RGBImage.convertTo(fImage, CV_32FC3); // for MXNet model
	split(fImage, m_InputImage);
    
    m_ArgsMap["data"] = NDArray(m_InputShape, m_CTX, false, 0);

	m_Executor->Forward(false);
	auto& outputs = m_Executor->outputs[0];

	std::vector<float> result;
	result.resize(2);
	
	for (unsigned int bIdx = 0; bIdx <1 ; bIdx++)
		outputs.Slice(bIdx, bIdx + 1).SyncCopyToCPU(&result, m_OutputShape[1]);

	std::vector<float> result_softmax = softmax(result);

	float output = result_softmax[0];
	return output;




    for(int i=0; i<inputs.size(); i++)
    {
       
    }



    for(int i=0; i<outputs.size(); i++)
    {
       
    }

    return Success;
}