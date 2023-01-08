#include "torch_inference.hpp"

torch::Device device = torch::kCPU;

void TorchEngine::tensorToDeepCTensor(torch::Tensor& torch_tensor, DeepCTensor& dc_tensor)
{
    auto tensor_shape = torch_tensor.sizes(); 
    for (int k = 0; k < tensor_shape.size(); k++)
        dc_tensor.shape.push_back((int)*(tensor_shape.data() + k));

    copyDataToTensor((float*)torch_tensor.data_ptr(), dc_tensor);
}

int TorchEngine::initialize(ConfigInfo setting)
{

    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(setting.model_file.c_str());
    }
    catch (const c10::Error &e)
    {
        return Fail_Init_Load_Model;
    }
    device = torch::cuda::is_available() && setting.use_gpu ? torch::kCUDA : torch::kCPU;
    //device = torch::kCPU;

    std::memcpy(this->norm_mean, setting.mean, sizeof(float) * 3);
    std::memcpy(this->norm_std, setting.std, sizeof(float) * 3);

    return Success;
}

int TorchEngine::inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor> &outputs)
{
    std::vector<torch::jit::IValue> input_tensors;
    for (int i = 0; i < inputs.size(); i++)
    {        
        int result_norm = this->normalize(inputs[i]);
        if (result_norm != Success)
        {
            return Fail_Infer_Set_Inputs;
        }
        auto tmp_input_tensor = torch::from_blob(inputs[i].data.data(), {inputs[i].shape[1], inputs[i].shape[2], inputs[i].shape[3]}, at::kFloat).to(device);
        tmp_input_tensor.unsqueeze_(0);
        input_tensors.push_back(tmp_input_tensor);  

        auto tmp_val = tmp_input_tensor.data_ptr<float>();
        auto tmp_shape = tmp_input_tensor.sizes();
    }
    
    // model forward
    auto out_tensor = module.forward(input_tensors);//.toTuple();

    // get outputs
    if(out_tensor.isTuple())    // multiple outputs
    {
        auto out_tensor_tuple = out_tensor.toTuple();
        for (int i = 0; i < outputs.size(); i++)
        {
            torch::Tensor tmp_output = out_tensor_tuple->elements()[i].toTensor();

            for (int j = 0; j < outputs.size(); j++)
            {
                if(outputs[j].name != tmp_output.name())
                    continue;

                tensorToDeepCTensor(tmp_output, outputs[j]);
            }
        }
    }
    else    // single output
    {
        torch::Tensor tmp_output = out_tensor.toTensor();
        auto tmp_val = tmp_output.data_ptr<float>();
        
        outputs.clear();
        DeepCTensor tmp_dc_tensor;
        tensorToDeepCTensor(tmp_output, tmp_dc_tensor);
        outputs.push_back(tmp_dc_tensor);
    }
 
    return Success;
}