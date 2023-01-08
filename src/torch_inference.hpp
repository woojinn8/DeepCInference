#ifndef __TORCH_INFERENCE_HPP__
#define __TORCH_INFERENCE_HPP__

#include "DeepCInference.hpp"

// Header for TORCH
#include <torch/script.h>
#include <torch/csrc/api/include/torch/cuda.h>

class TorchEngine : public DeepCInference_Vision
{

public:
  TorchEngine(){};
  ~TorchEngine(){};

  int initialize(ConfigInfo setting);
  int inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor> &outputs);

private:
  torch::jit::script::Module module;
  
  void tensorToDeepCTensor(torch::Tensor& torch_tensor, DeepCTensor& dc_tensor);
};
#endif