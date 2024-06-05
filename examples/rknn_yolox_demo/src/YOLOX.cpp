#include <stdio.h>
#include <sys/time.h>
#include <opencv2/imgproc.hpp>
#include "YOLOX.hpp"

const char* coco_80_labels[] = {
#include "../model/coco_80_labels.h"
};

YOLOX::YOLOX(): RKNN(), _channel(3), _inputs(nullptr), _outputs(nullptr), _scale_x(1), _scale_y(1)
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

        printf("%d outputs\n", _io_num.n_output);
        _outputs = new rknn_output[_io_num.n_output];
        for (int i = 0; i < _io_num.n_output; i++)
        {
            memset(&_outputs[i], 0, sizeof(rknn_output));
            _outputs[i].want_float = 0;
        }
        return true;
    }
    return false;
}

static double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

cv::Mat YOLOX::Infer(cv::Mat& image)
{
    int img_width = image.cols;
    int img_height = image.rows;
    timeval start, stop;

    if (_width == img_width && _height == img_height)
    {
        cv::cvtColor(image, _image, cv::COLOR_BGR2RGB);
    }
    else
    {
        _scale_x = (float)_width / img_width;
        _scale_y = (float)_height / img_height;
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(_width, _height));
        cv::cvtColor(resized, _image, cv::COLOR_BGR2RGB);
    }
    _inputs[0].buf = _image.data;

    gettimeofday(&start, NULL);
    // Model inference
    int ret = rknn_run(_ctx, NULL);
    //printf("rknn_run returns %d\n", ret);
    ret = rknn_outputs_get(_ctx, _io_num.n_output, _outputs, NULL);
    //printf("rknn_outputs_get returns %d\n", ret);
    gettimeofday(&stop, NULL);
    printf("once run use %f ms\n", (__get_us(stop) - __get_us(start)) / 1000);
    
    PostProcess();
    
    return image;
}

static int strides[] = {8, 16, 32};
static int anchor0[6] = {10, 13, 16, 30, 33, 23};
static int anchor1[6] = {30, 61, 62, 45, 59, 119};
static int anchor2[6] = {116, 90, 156, 198, 373, 326};

void YOLOX::PostProcess()
{
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i < _io_num.n_output; ++i)
    {
        out_scales.push_back(_output_attrs[i].scale);
        out_zps.push_back(_output_attrs[i].zp);
    }

    GenerateProposals((int8_t*)_outputs[0].buf, anchor0, strides[0], out_zps[0], out_scales[0]);
    GenerateProposals((int8_t*)_outputs[1].buf, anchor1, strides[1], out_zps[1], out_scales[1]);    
    GenerateProposals((int8_t*)_outputs[2].buf, anchor2, strides[2], out_zps[2], out_scales[2]);   

    if (2 <= _proposals.size())
    {
        std::sort(_proposals.begin(), _proposals.end());
    }
    std::vector<int> picked = nmsSortedBoxes();
    _objects.resize(picked.size());
    for (size_t i = 0; i < picked.size(); i++)
    {
        _objects[i] = _proposals[picked[i]];

        _objects[i].box.left /= _scale_x;
        _objects[i].box.top /= _scale_y;
        _objects[i].box.right /= _scale_x;
        _objects[i].box.bottom /= _scale_x;
    }
}

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) 
{ 
    return ((float)qnt - (float)zp) * scale; 
}

void YOLOX::GenerateProposals(int8_t *input, int *anchor, int stride, int zp, float scale)
{
    int grid_w = _width / stride;
    int grid_h = _height / stride;
    int grid_len = grid_h * grid_w;
    int8_t thres_i8 = qnt_f32_to_affine(_box_conf_threshold, zp, scale);
    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (thres_i8 <= box_confidence)
                {
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = input + offset;
                    float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                    {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (thres_i8 <= maxClassProbs)
                    {
                        OBJECT proposal;
                        proposal.id = maxClassId;
                        proposal.prob = (deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_to_f32(box_confidence, zp, scale));
                        proposal.box = {box_x, box_y, box_x + box_w, box_y + box_h};
                        _proposals.push_back(proposal);
                    }
                }
            }
        }
    }
}

std::vector<int> YOLOX::nmsSortedBoxes()
{
    std::vector<int> picked;
}