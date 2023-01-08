#include "ncnn_inference.hpp"

int NCNNEngine::initialize(ConfigInfo setting)
{
    model_bin = setting.model_file;
    model_param = setting.param_file;
    num_thread = setting.num_thread;
    use_gpu = setting.use_gpu;
    use_fp16 = setting.use_fp16;
    power_save = setting.powersave;


#ifdef VULKAN_BUILD
    if (use_gpu)
    {
        if (ncnn::get_gpu_count() > 0)
        {
            net.opt.use_vulkan_compute = 1;
        }
    }
#endif
    if (use_fp16)
    {
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = true;
    }
    else
    {
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
    }

    int load_result_param = net.load_param(model_param.c_str());
    int load_result_bin = net.load_model(model_bin.c_str());

    if (load_result_param != 0 || load_result_bin != 0)
    {
        printf("Fail to load model : param(%d), bin(%d)\n", load_result_param, load_result_bin);
        return Fail_Init_Load_Model;
    }

    std::memcpy(this->norm_mean, setting.mean, sizeof(float)*3);
    std::memcpy(this->norm_std, setting.std, sizeof(float)*3);

    return Success;
}


int NCNNEngine::inference(std::vector<DeepCTensor> inputs, std::vector<DeepCTensor>& outputs)
{

    ncnn::Extractor ex = net.create_extractor();
#ifdef VULKAN_BUILD
    if (use_gpu)
    {
        ex.set_vulkan_compute(true);
    }
#endif

    ex.set_num_threads(num_thread);

    ncnn::CpuSet ps_ = ncnn::get_cpu_thread_affinity_mask(power_save);
    ncnn::set_cpu_thread_affinity(ps_);

    for(int i=0; i<inputs.size(); i++)
    {
        int width_input, height_input;
        if(inputs[i].storedType == StoredType::BHWC)
        {
            width_input = inputs[i].shape[2];
            height_input = inputs[i].shape[1];
        }
        else
        {
            width_input = inputs[i].shape[3];
            height_input = inputs[i].shape[2];
        }

        ncnn::Mat in = ncnn::Mat::from_pixels(inputs[0].img.data, ncnn::Mat::PIXEL_BGR2RGB, width_input, height_input);
        in.substract_mean_normalize(this->norm_mean, this->norm_std);

        ex.input(inputs[i].name.c_str(), in);
    }

    for(int i=0; i<outputs.size(); i++)
    {
        ncnn::Mat res_ori;
        ex.extract(outputs[i].name.c_str(), res_ori);

        outputs[i].shape.clear();
        outputs[i].shape.push_back(res_ori.c);
        outputs[i].shape.push_back(res_ori.h);
        outputs[i].shape.push_back(res_ori.w);

        copyDataToTensor(&res_ori[0], outputs[i]);
    }

    return Success;
}