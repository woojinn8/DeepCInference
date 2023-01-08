#ifndef __TFLITE_INFERENCE_HPP__
#define __TFLITE_INFERENCE_HPP__

#include "DeepCInference.hpp"

// Header for tflite C++ API
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

// Header for tflite C aPI
#include "tensorflow/lite/c/c_api.h"

// Header for GPU delegate
#include "tensorflow/lite/delegates/gpu/delegate.h"

using namespace tflite;

// C++ API
class TfliteCppEngine : public DeepCInference_Vision
{
public:
  TfliteCppEngine(){};
  ~TfliteCppEngine()
  {
    model.release();
    interpreter.release();
  };

  int initialize(ConfigInfo setting);
  int inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor>& outputs);

private:
  
  std::unique_ptr<tflite::FlatBufferModel> model;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter; 
  
  TfLiteDelegate *delegate;

  int num_interpreter_inputs, num_interpreter_outputs;
  int output_channel, output_height, output_width;
};


// C API
class TfliteCEngine : public DeepCInference_Vision
{
public:
  TfliteCEngine(){};
  ~TfliteCEngine()
  {
    TfLiteInterpreterDelete(this->interpreter);
    TfLiteGpuDelegateV2Delete(this->delegate);
    TfLiteInterpreterOptionsDelete(this->options);
    TfLiteModelDelete(this->model);
  };
  int initialize(ConfigInfo setting);
  int inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor>& outputs);

private:
  TfLiteModel *model;
  TfLiteInterpreterOptions *options;
  TfLiteInterpreter *interpreter;

  TfLiteDelegate *delegate;

  int output_channel, output_height, output_width;
};

#endif