#include <stdio.h>
#include <opencv2/imgproc.hpp>
#include "YOLOX.hpp"

YOLOX::YOLOX(): RKNN(), _channel(3), _inputs(nullptr), _outputs(nullptr)
{

}

YOLOX::~YOLOX()
{
    if (_inputs != nullptr)
        delete[] _inputs;
    if (_outputs != nullptr)
        delete[] _outputs;
    _inputs = nullptr;
    _outputs = nullptr;
}

bool YOLOX::Initialize(const char* model_filepath, float nms_threshold, float box_threshold, rknn_core_mask core_mask)
{
    if (RKNN::Initialize(model_filepath, core_mask))
    {
        _nms_threshold = nms_threshold;
        _box_conf_threshold = box_threshold;

        if (_input_attrs[0].fmt == RKNN_TENSOR_NCHW)
        {
            printf("model is NCHW input fmt\n");
            _channel = _input_attrs[0].dims[1];
            _height = _input_attrs[0].dims[2];
            _width = _input_attrs[0].dims[3];
        }
        else
        {
            printf("model is NHWC input fmt\n");
            _height = _input_attrs[0].dims[1];
            _width = _input_attrs[0].dims[2];
            _channel = _input_attrs[0].dims[3];
        }
        printf("model input height=%d, width=%d, channel=%d\n", _height, _width, _channel);

        _inputs = new rknn_input[_io_num.n_input];
        _inputs[0].index = 0;
        _inputs[0].type = RKNN_TENSOR_UINT8;
        _inputs[0].size = _width * _height * _channel;
        _inputs[0].fmt = RKNN_TENSOR_NHWC;
        _inputs[0].pass_through = 0;
        _image.create(_height, _width, CV_8UC3);

        _outputs = new rknn_output[_io_num.n_output];
        for (int i = 0; i < _io_num.n_output; i++)
        {
            _outputs[i].want_float = 0;
        }
        return true;
    }
    return false;
}

cv::Mat YOLOX::Infer(cv::Mat& image)
{
    int img_width = image.cols;
    int img_height = image.rows;
    if (_width == img_width && _height == img_height)
    {
        cv::cvtColor(image, _image, cv::COLOR_BGR2RGB);
    }
    else
    {
        float scale_w = (float)_width / img_width;
        float scale_h = (float)_height / img_height;
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(_width, _height));
        cv::cvtColor(resized, _image, cv::COLOR_BGR2RGB);
    }
    _inputs[0].buf = _image.data;

    // Model inference
    int ret = rknn_run(_ctx, NULL);
    ret = rknn_outputs_get(_ctx, _io_num.n_output, _outputs, NULL);

    PostProcess();
    
    return image;
}

std::vector<OBJECT> YOLOX::PostProcess()
{
    std::vector<OBJECT> objects;

    return objects;
}