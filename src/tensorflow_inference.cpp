#include "tensorflow_inference.hpp"

int TensorflowEngine::initialize(ConfigInfo setting)
{   
    if(setting.use_gpu)
        setenv("CUDA_VISIBLE_DEVICES", "", 0); 

    session = TF_LoadSessionFromSavedModel(session_options, run_options,
                                           setting.model_file.c_str(), tags.data(), tags.size(),
                                           graph, nullptr, status);
    if (TF_GetCode(status) != TF_OK)
    {
        std::cout << TF_Message(status) << '\n';
        return Fail_Init_Load_Model;
    }

    

    std::memcpy(this->norm_mean, setting.mean, sizeof(float)*3);
    std::memcpy(this->norm_std, setting.std, sizeof(float)*3);

    return Success;
}

int TensorflowEngine::inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor>& outputs)
{
    // set input op
    std::vector<TF_Output> input_TF_Outputs;
    std::vector<TF_Tensor*> input_values;
    for(int i=0; i<inputs.size(); i++)
    {
        TF_Operation *input_op = TF_GraphOperationByName(graph, inputs[i].name.c_str());
        
        if (input_op == nullptr) 
            return Fail_Init_Set_Input_Output;
        
        TF_Output input_output = {input_op, 0};
        input_TF_Outputs.push_back(input_output);

        int num_dims = inputs[i].shape.size();
        std::vector<int64_t> in_dims(num_dims);

        size_t num_bytes_in = sizeof(float);
        for (int j = 0; j < inputs[i].shape.size(); j++)
        {   
            in_dims[j] = inputs[i].shape[j];
            num_bytes_in *= (size_t)inputs[i].shape[j];
        }

        int result_norm = this->normalize(inputs[i]);
        if (result_norm != Success)
        {
            printf("Fail to normalize Tensor : %d\n", result_norm);
            return -1;
        }

        auto const deallocator = [](void *, std::size_t, void *) {}; // unused deallocator because of RAII
        auto* input_tensor = TF_NewTensor(TF_FLOAT, in_dims.data(), num_dims, inputs[i].data.data(), num_bytes_in, deallocator, nullptr);
        input_values.push_back(input_tensor);
    }

    // set output op
    std::vector<TF_Output> output_TF_Output;
    std::vector<TF_Tensor*> output_values;
    for(int i=0; i<outputs.size(); i++)
    {
       TF_Operation* output_op = TF_GraphOperationByName(graph,outputs[i].name.c_str());
       if (output_op == nullptr) 
            return Fail_Init_Set_Input_Output;

       TF_Output output_opout = {output_op,0};
       output_TF_Output.push_back(output_opout);
       
       int num_dims = outputs[i].shape.size();
       std::vector<int64_t> out_dims(num_dims);

       size_t num_bytes_out = sizeof(float);
       for (int j = 0; j < outputs[i].shape.size(); j++)
       {
           out_dims[j] = outputs[i].shape[j];
           num_bytes_out *= (size_t)outputs[i].shape[j];
       }

       TF_Tensor *output_value = TF_AllocateTensor(TF_FLOAT, out_dims.data(), num_dims, num_bytes_out);
       output_values.push_back(output_value);
       
    }
    
    // Run
    TF_SessionRun(session,
                  run_options,
                  input_TF_Outputs.data(), input_values.data(), input_TF_Outputs.size(),
                  output_TF_Output.data(), output_values.data(), output_TF_Output.size(),
                  nullptr, 0,
                  nullptr,
                  status);
    if (TF_GetCode(status) != TF_OK)
    {
        std::cout << TF_Message(status) << std::endl;
        return Fail_Infer_Invoke;
    }

    // get output 
    for(int i=0; i<outputs.size(); i++)
    {
        outputs[i].storedType = StoredType::None;

        int num_dim = TF_NumDims(output_values[i]);
        for (int j = 0; j < num_dim; j++)
        {
           outputs[i].shape.push_back(TF_Dim(output_values[i], j));
        }

        const auto data = static_cast<float *>(TF_TensorData(output_values[i]));
        copyDataToTensor(data, outputs[i]);
    }

    for(int i=0; i<inputs.size(); i++)
        TF_DeleteTensor(input_values[i]);

    for(int i=0; i<outputs.size(); i++)
        TF_DeleteTensor(output_values[i]);

    return Success;
}