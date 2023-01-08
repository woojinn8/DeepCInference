#include "DeepCInference.hpp"

#include <filesystem>

char *current_path;

void getTensorflowSetting(ConfigInfo& setting)
{
    setting.model_file = std::string(current_path) + "/bin/linux/mobilenetv2";
    setting.num_thread = 1;
    setting.use_gpu = 0;
    setting.use_fp16 = 1;

    setting.mean[0] = 0.485 * 255.0;
    setting.mean[1] = 0.456 * 255.0;
    setting.mean[2] = 0.406 * 255.0;
    setting.std[0] = 0.229 * 255.0;
    setting.std[1] = 0.224 * 255.0;
    setting.std[2] = 0.225 * 255.0;    

    setting.input_node_name.push_back("serving_default_input");
    setting.output_node_name.push_back("PartitionedCall");  
};


void getTfliteSetting(ConfigInfo& setting)
{
    setting.model_file = std::string(current_path) + "/bin/linux/mobilenetv2.tflite";
    setting.num_thread = 1;
    setting.use_gpu = 0;
    setting.use_fp16 = 1;

    setting.mean[0] = 0.485 * 255.0;
    setting.mean[1] = 0.456 * 255.0;
    setting.mean[2] = 0.406 * 255.0;
    setting.std[0] = 0.229 * 255.0;
    setting.std[1] = 0.224 * 255.0;
    setting.std[2] = 0.225 * 255.0;    

    setting.input_node_name.push_back("serving_default_input:0");
    setting.output_node_name.push_back("PartitionedCall:0");  
};

void getTorchSetting(ConfigInfo& setting)
{
    setting.model_file = std::string(current_path) + "/bin/linux/mobilenetv2.pt";
    setting.num_thread = 1;
    setting.use_gpu = 1;
    setting.use_fp16 = 1;

    setting.mean[0] = 0.485 * 255.0;
    setting.mean[1] = 0.456 * 255.0;
    setting.mean[2] = 0.406 * 255.0;
    setting.std[0] = 0.229 * 255.0;
    setting.std[1] = 0.224 * 255.0;
    setting.std[2] = 0.225 * 255.0;  

    setting.input_node_name.push_back("input");
    setting.output_node_name.push_back("output");
}

void getNCNNSetting(ConfigInfo& setting)
{
    setting.model_file = std::string(current_path) + "/bin/linux/mobilenetv2.bin";
    setting.param_file = std::string(current_path) + "/bin/linux/mobilenetv2.param";
    setting.num_thread = 1;
    setting.use_gpu = 0;
    setting.use_fp16 = 1;

    setting.mean[0] = 0.485 * 255.0;
    setting.mean[1] = 0.456 * 255.0;
    setting.mean[2] = 0.406 * 255.0;
    setting.std[0] = 1/(0.229 * 255.0);
    setting.std[1] = 1/(0.224 * 255.0);
    setting.std[2] = 1/(0.225 * 255.0);  

    setting.input_node_name.push_back("input");
    setting.output_node_name.push_back("output");
};

void getONNXSetting(ConfigInfo& setting)
{
    setting.model_file = std::string(current_path) + "/bin/linux/mobilenetv2.onnx";
    setting.num_thread = 1;
    setting.use_gpu = 0;
    setting.use_fp16 = 1;

    setting.mean[0] = 0.485 * 255.0;
    setting.mean[1] = 0.456 * 255.0;
    setting.mean[2] = 0.406 * 255.0;
    setting.std[0] = 0.229 * 255.0;
    setting.std[1] = 0.224 * 255.0;
    setting.std[2] = 0.225 * 255.0;  

    setting.input_node_name.push_back("input");
    setting.output_node_name.push_back("output");
};

void getMXNetSetting(ConfigInfo& setting)
{
    setting.model_file = std::string(current_path) + "/bin/linux/mobilenetv2-models.sym";
    setting.param_file = std::string(current_path) + "/bin/linux/mobilenetv2-0001.params";
    setting.num_thread = 1;
    setting.use_gpu = 0;
    setting.use_fp16 = 1;

    setting.mean[0] = 0.485 * 255.0;
    setting.mean[1] = 0.456 * 255.0;
    setting.mean[2] = 0.406 * 255.0;
    setting.std[0] = 0.229 * 255.0;
    setting.std[1] = 0.224 * 255.0;
    setting.std[2] = 0.225 * 255.0;  

    setting.input_node_name.push_back("input");
    setting.output_node_name.push_back("output");
}

void getMNNSetting(ConfigInfo& setting)
{
    setting.model_file = std::string(current_path) + "/bin/linux/mobilenetv2.mnn";
    setting.num_thread = 1;
    setting.use_gpu = 0;
    setting.use_fp16 = 1;

    setting.mean[0] = 0.485 * 255.0;
    setting.mean[1] = 0.456 * 255.0;
    setting.mean[2] = 0.406 * 255.0;
    setting.std[0] = 1/(0.229 * 255.0);
    setting.std[1] = 1/(0.224 * 255.0);
    setting.std[2] = 1/(0.225 * 255.0);   

    setting.input_node_name.push_back("data");
    setting.output_node_name.push_back("mobilenetv20_output_flatten0_reshape0");
};

void getOpenCVDNNSetting(ConfigInfo& setting)
{
    setting.model_file = std::string(current_path) + "/bin/linux/mobilenetv2.onnx";
    setting.num_thread = 1;
    setting.use_gpu = 0;
    setting.use_fp16 = 1;

    setting.mean[0] = 0.485 * 255.0;
    setting.mean[1] = 0.456 * 255.0;
    setting.mean[2] = 0.406 * 255.0;
    setting.std[0] = 0.229 * 255.0;
    setting.std[1] = 0.224 * 255.0;
    setting.std[2] = 0.225 * 255.0;  

    setting.input_node_name.push_back("input");
    setting.output_node_name.push_back("output");
};


std::vector<float> softmax(std::vector<float> arr)
{	
	float maxElement = *std::max_element(arr.begin(), arr.end());
	std::vector<float> result;
    float sum = 0.0;
	for(auto const& i : arr)
    { 
        float val = std::exp(i - maxElement);
        sum += val;
        result.push_back(val);
    }
	
    DivVectorByScalar(result, sum, 0, arr.size());
    return result;
};


int main(int argc, char** argv)
{ 
    int framework = 0;
    if(argc > 1)
        framework = std::atoi(argv[1]);

    current_path = get_current_dir_name();
    printf("0. Start at %s\n", current_path);
    std::shared_ptr<DeepCInference_Vision> dci = nullptr;


    printf("1. Create Engine\n");
    FrameworkType framework_type = (FrameworkType)framework;
    dci = DeepCInference_Vision::create(framework_type);
    

    printf("2. Set Config\n");
    ConfigInfo setting;
    switch(framework_type)
    {   
        case FrameworkType::type_tensorflow: // 0
            getTensorflowSetting(setting);
            break;

        case FrameworkType::type_tflite_c:   // 1
        case FrameworkType::type_tflite_cpp :    // 2
            getTfliteSetting(setting);
        break;

        case FrameworkType::type_torch : // 3
            getTorchSetting(setting);
        break;

        case FrameworkType::type_ncnn:   // 4
            getNCNNSetting(setting);
        break;

        case FrameworkType::type_onnx:   // 5
            getONNXSetting(setting);
        break;

        case FrameworkType::type_mxnet:    // 6
            //getMXNetSetting(setting);
        break;

        case FrameworkType::type_mnn:    // 7
            getMNNSetting(setting);
        break;

        case FrameworkType::type_opencv_dnn:    // 8
            getOpenCVDNNSetting(setting);
        break;
        
    }
        

    printf("3. Initialize\n");
    int result_init = dci->initialize(setting);
    if(result_init != Success)
    {
        printf("Fail to initialize : %d\n", result_init);
        return -1;
    }


    printf("4. Prepare Sample image\n");
    std::string sample_img_path = std::string(current_path) + "/bin/n01443537_goldfish.JPEG";
    cv::Mat sample_img = cv::imread(sample_img_path);
    if(sample_img.empty())
    {   
        printf("Invalid input image\n"); 
        return -1;
    }
    cv::Mat sample_resize;
    cv::resize(sample_img, sample_resize, cv::Size(224,224));   // resize for imagenet size


    printf("5. Prepare Input Tensor Vector for Inference\n");
    std::vector<DeepCTensor> inputs, outputs;

    DeepCTensor tmp_in_tensor;
    int result_createTensor = dci->createTensor(sample_resize, tmp_in_tensor, setting.input_node_name[0]);
    if (result_createTensor != Success)
    {
        printf("Fail to create Tensor : %d\n", result_createTensor);
        return -1;
    }
    
    inputs.push_back(tmp_in_tensor);

    cv::Mat empty_mat;
    DeepCTensor tmp_out_tensor;
    result_createTensor = dci->createTensor(empty_mat, tmp_out_tensor, setting.output_node_name[0]);
    if (result_createTensor != Success)
    {
        printf("Fail to create Tensor : %d\n", result_createTensor);
        return -1;
    }
    outputs.push_back(tmp_out_tensor);


    printf("6. Do Inference\n");
    int result_infer = dci->inference(inputs, outputs);
    if(result_infer != Success)
    {
        printf("Fail to Inference : %d\n", result_infer);
        return -1;
    }

    if(outputs.size() == 0)
    {
        printf("Invalid output : %ld\n", outputs.size());
        return -1;
    }


    printf("8. Output result -> shape : (%d", outputs[0].shape[0]);
    for(int i=1; i<outputs[0].shape.size(); i++)
        printf(",%d", outputs[0].shape[i]);

    std::vector<float> output_val = softmax(outputs[0].data);
    printf("), output_result : %f, %f, %f\n", output_val[0], output_val[1], output_val[2]);

    outputs.clear();


    printf("9. Complete Inference\n");
    return 0;
}