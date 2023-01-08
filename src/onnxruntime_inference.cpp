#include "onnxruntime_inference.hpp"

int ONNXRuntimeCppEngine::initialize(ConfigInfo setting)
{
    // onnxruntime setup
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
    Ort::SessionOptions session_options;
    session_options.SetInterOpNumThreads(setting.num_thread);
    session_options.SetIntraOpNumThreads(setting.num_thread);

    if(setting.use_gpu)
    {
        OrtCUDAProviderOptions cuda_options{};
        //cuda_options.device_id = 0; //single gpu only
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
    // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    //session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    this->session = std::make_shared<Ort::Experimental::Session>(env, setting.model_file, session_options);

    // name/shape of inputs
    input_names = session->GetInputNames();
    input_shapes = session->GetInputShapes();
    Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

    // name/shape of outputs
    output_names = session->GetOutputNames();
    output_shapes = session->GetOutputShapes();
    Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    
    std::memcpy(this->norm_mean, setting.mean, sizeof(float)*3);
    std::memcpy(this->norm_std, setting.std, sizeof(float)*3);

    return Success;
}

int ONNXRuntimeCppEngine::inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor> &outputs)
{
    // Set input
    if (inputs.size() == 0)
      return Fail_Infer_Set_Inputs;

    // Set Input
    std::vector<Ort::Value> input_tensors;
    for (int i = 0; i < input_names.size(); i++)
    {
        auto input_shape = input_shapes[i];
        
        if(input_shape[0] < 1)
            input_shape[0] = 1;

        for(int j=0; j<inputs.size(); j++)
        {
            if(input_names[i] != inputs[j].name)
                continue;

            int result_norm = this->normalize(inputs[j]);
            if (result_norm != Success)
            {
                printf("Fail to normalize Tensor : %d\n", result_norm);
                return -1;
            }
            
            input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(inputs[j].data.data(), inputs[j].data.size(), input_shape));
            // double-check the dimensions of the input tensor
            if(!input_tensors[i].IsTensor() || input_tensors[i].GetTensorTypeAndShapeInfo().GetShape() != input_shape)
                return Fail_Infer_Set_Inputs;
        }
    }

    // Run
    Ort::RunOptions run_options;
    std::vector<Ort::Value> output_tensors = session->Run(input_names, input_tensors, output_names, Ort::RunOptions{nullptr}) ;
    //for (size_t i = 0; i < output_names.size(); i++) output_tensors.emplace_back(nullptr);
    //session->Run(input_names, input_tensors, output_names, output_tensors, Ort::RunOptions{nullptr});

    // Get output
    for(int i=0; i<output_tensors.size(); i++)
    {   
        for(int j=0; j<outputs.size(); j++)
        {
            if(outputs[j].name != output_names[i])
                continue;

            outputs[j].storedType = StoredType::None;
            float* floatarr = output_tensors[i].GetTensorMutableData<float>();
            std::vector<int64_t> size_o = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();

            for (int k = 0; k < size_o.size(); k++)
                outputs[j].shape.push_back((int)size_o[k]);
            copyDataToTensor(floatarr, outputs[j]);
        }
    }
    
    return Success;    
}
