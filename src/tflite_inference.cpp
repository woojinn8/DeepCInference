#include "tflite_inference.hpp"

int TfliteCppEngine::initialize(ConfigInfo setting)
{
  try
  {
    this->model = tflite::FlatBufferModel::BuildFromFile(setting.model_file.c_str());
    tflite::InterpreterBuilder(*this->model, this->resolver)(&this->interpreter);

    int result_delegate;
    if (setting.use_gpu)
    {
      auto options = TfLiteGpuDelegateOptionsV2Default();
      options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
      if (setting.use_fp16)
      {
        options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        options.is_precision_loss_allowed = 1;
      }
      else
      {
        options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
      }
      this->delegate = TfLiteGpuDelegateV2Create(&options);
      result_delegate = this->interpreter->ModifyGraphWithDelegate(this->delegate);
      if (result_delegate != 0)
        return Fail_Init_Set_GPU;
    }

    int result_set_thread = this->interpreter->SetNumThreads(setting.num_thread);
    if (result_set_thread != 0)
      return Fail_Init_Set_Thread;

    int result_allocate = this->interpreter->AllocateTensors();
    if (result_allocate != 0)
      return Fail_Init;

    num_interpreter_inputs = this->interpreter->inputs().size();
    num_interpreter_outputs = this->interpreter->outputs().size();

    // Warm-up
    for (int i = 0; i < 3; i++)
      this->interpreter->Invoke();
  }
  catch (int expn)
  {
    std::cout << "Fail to Initialize : " << expn << std::endl;
    return Fail_Init;
  }

  std::memcpy(this->norm_mean, setting.mean, sizeof(float) * 3);
  std::memcpy(this->norm_std, setting.std, sizeof(float) * 3);

  return Success;
}

int TfliteCppEngine::inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor> &outputs)
{
  try
  {
    // Set input
    if (inputs.size() != num_interpreter_inputs)
      return Fail_Infer_Set_Inputs;

    for (int i = 0; i < num_interpreter_inputs; i++)
    {
      float *input = interpreter->typed_input_tensor<float>(i);
      std::string input_node_name = interpreter->GetInputName(i);

      for (int j = 0; j < inputs.size(); j++)
      {
        if (inputs[j].name != input_node_name) // check tensor_name == graph_node_name correct?
          continue;

        int result_preprocess = normalize(inputs[j]);
        if (result_preprocess != Success)
        {
          printf("Fail to preprocess : %d\n", result_preprocess);
          return -1;
        }

        std::memcpy(input, inputs[j].data.data(), sizeof(float) * getTensorSize(inputs[j]));
      }
    }

    // Inference
    this->interpreter->Invoke();

    // Get output
    outputs.clear();
    TfLiteTensor *output_tensor = nullptr;

    for (int i = 0; i < num_interpreter_outputs; i++)
    {
      output_tensor = this->interpreter->output_tensor(i);

      DeepCTensor tmp_tensor;
      tmp_tensor.name = TfLiteTensorName(output_tensor);
      tmp_tensor.storedType = StoredType::None;

      int num_dim = TfLiteTensorNumDims(output_tensor);
      for (int j = 0; j < num_dim; j++)
      {
        tmp_tensor.shape.push_back(TfLiteTensorDim(output_tensor, j));
      }

      float *output = interpreter->typed_output_tensor<float>(i);

      copyDataToTensor(output, tmp_tensor);
      outputs.push_back(tmp_tensor);
    }
  }
  catch (int expn)
  {
    std::cout << "Fail to Inference : " << expn << std::endl;
    return Fail_Infer;
  }

  return Success;
}

int TfliteCEngine::initialize(ConfigInfo setting)
{
  this->model = TfLiteModelCreateFromFile(setting.model_file.c_str());
  this->options = TfLiteInterpreterOptionsCreate();

  if (setting.use_gpu)
  {
    auto options_gpu = TfLiteGpuDelegateOptionsV2Default();

    if (setting.use_fp16)
    {
      options_gpu.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
      options_gpu.is_precision_loss_allowed = 1;
    }
    else
    {
      options_gpu.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
    }

    this->delegate = TfLiteGpuDelegateV2Create(&options_gpu);
    TfLiteInterpreterOptionsAddDelegate(this->options, this->delegate);
  }

  TfLiteInterpreterOptionsSetNumThreads(this->options, setting.num_thread);

  this->interpreter = TfLiteInterpreterCreate(this->model, this->options);

  TfLiteStatus tfresult;
  tfresult = TfLiteInterpreterAllocateTensors(this->interpreter);
  if (tfresult != kTfLiteOk)
  {
    std::cout << "Fail to allocate tensor" << std::endl;
    return Fail_Init;
  }

  // warm-up
  TfLiteTensor *input_tensor = TfLiteInterpreterGetInputTensor(this->interpreter, 0);

  for (int i = 0; i < 3; i++)
  {
    tfresult = TfLiteInterpreterInvoke(this->interpreter);
    if (tfresult != kTfLiteOk)
    {
      std::cout << "Fail to Invoke" << std::endl;
      return Fail_Init;
    }
  }

  std::memcpy(this->norm_mean, setting.mean, sizeof(float) * 3);
  std::memcpy(this->norm_std, setting.std, sizeof(float) * 3);

  return Success;
}

int TfliteCEngine::inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor> &outputs)
{
  try
  {
    TfLiteStatus tfresult;
    // Set input
    if (inputs.size() == 0)
      return Fail_Infer_Set_Inputs;

    // Set Input
    for (int i = 0; i < inputs.size(); i++)
    {
      TfLiteTensor *input_tensor = TfLiteInterpreterGetInputTensor(this->interpreter, i);
      std::string input_node_name = TfLiteTensorName(input_tensor);
      for (int j = 0; j < inputs.size(); j++)
      {
        if (inputs[j].name != input_node_name) // check tensor_name == graph_node_name correct?
          continue;

        int result_norm = this->normalize(inputs[j]);
        if (result_norm != Success)
        {
          printf("Fail to normalize Tensor : %d\n", result_norm);
          return -1;
        }

        TfLiteTensorCopyFromBuffer(input_tensor, inputs[j].data.data(), sizeof(float) * getTensorSize(inputs[j]));
      }
    }

    // Inference
    tfresult = TfLiteInterpreterInvoke(this->interpreter);
    if (tfresult != kTfLiteOk)
    {
      std::cout << "Fail to invoke" << std::endl;
      return Fail_Infer;
    }

    // Get output
    outputs.clear();
    const TfLiteTensor *output_tensor = nullptr;
    int num_output = TfLiteInterpreterGetOutputTensorCount(this->interpreter);
    for (int i = 0; i < num_output; i++)
    {
      output_tensor = TfLiteInterpreterGetOutputTensor(this->interpreter, i);

      DeepCTensor tmp_tensor;
      tmp_tensor.name = TfLiteTensorName(output_tensor);
      tmp_tensor.storedType = StoredType::None;

      int num_dim = TfLiteTensorNumDims(output_tensor);
      for (int j = 0; j < num_dim; j++)
      {
        tmp_tensor.shape.push_back(TfLiteTensorDim(output_tensor, j));
      }

      float *output = new float[getTensorSize(tmp_tensor)];
      TfLiteTensorCopyToBuffer(output_tensor, output, sizeof(float) * getTensorSize(tmp_tensor));
      copyDataToTensor(output, tmp_tensor);
      outputs.push_back(tmp_tensor);
      delete[] output;
    }
  }
  catch (int expn)
  {
    std::cout << "Fail to Inference : " << expn << std::endl;
    return Fail_Infer;
  }
  return Success;
}
