#include <stdio.h>
#include <sys/time.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "im2d.h"
#include "YOLOv5.hpp"

#define USE_LETTERBOX

const char* coco_80_labels[] = {
#include "../model/coco_80_labels.h"
};

static int strides[] = {8, 16, 32};
static int anchor0[6] = {10, 13, 16, 30, 33, 23};
static int anchor1[6] = {30, 61, 62, 45, 59, 119};
static int anchor2[6] = {116, 90, 156, 198, 373, 326};

YOLOv5::YOLOv5(): RKNN(), _channel(3), _inputs(nullptr), _outputs(nullptr), _scale_x(1), _scale_y(1)
{

}

YOLOv5::~YOLOv5()
{
    if (_inputs != nullptr)
        delete[] _inputs;
    if (_outputs != nullptr)
        delete[] _outputs;
    _inputs = nullptr;
    _outputs = nullptr;
}

bool YOLOv5::Initialize(const char* model_filepath, float nms_threshold, float box_threshold, rknn_core_mask core_mask)
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
        printf("Generate Grids and Strides %d x %d\n", _width, _height);
        // generate Grids and Strides
        for (int i = 0; i < 3; i++)
        {
            int y_grids = _height / strides[i];
            int x_grids = _width / strides[i];
            for (int y = 0; y < y_grids; y++)
            {
                GridAndStride grid = {strides[i], 0, y};
                for (int x = 0; x < x_grids; x++)
                {
                    grid.x = x;
                    _grids[i].push_back(grid);
                }
            }
            printf("grid:%ld\n", _grids[i].size());
        }        
        return true;
    }
    return false;
}

static double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

cv::Mat YOLOv5::Infer(cv::Mat& image)
{
    timeval start, stop;
    PreProcess(image);

    gettimeofday(&start, NULL);
    // Model inference
    int ret = rknn_run(_ctx, NULL);
    
    ret = rknn_outputs_get(_ctx, _io_num.n_output, _outputs, NULL);

    gettimeofday(&stop, NULL);
    printf("once run use %f ms\n", (__get_us(stop) - __get_us(start)) / 1000);
    
    PostProcess();

    return image;
}

bool YOLOv5::PreProcess(cv::Mat& image)
{
#ifdef USE_LETTERBOX
    float scale = image.rows > image.cols? (float)_height / image.rows: (float)_width / image.cols;

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(), scale, scale);

    int top = (_height - resized_image.rows) / 2;
    int left = (_width - resized_image.cols) / 2;
    printf("Scale:%f top:%d left:%d\n", scale, top, left);
    // Cast resized image
    cv::copyMakeBorder(resized_image, _image, top, top, left, left, cv::BORDER_CONSTANT, cv::Scalar(128, 0, 0));
    cv::imwrite("letterbox.png", _image);
    _inputs[0].buf = _image.data;
    return true;
 #else
    int img_width = image.cols; 
    int img_height = image.rows;

    im_rect src_rect = {0};
    im_rect dst_rect = {0};
    rga_buffer_t src = wrapbuffer_virtualaddr((void *)image.data, img_width, img_height, RK_FORMAT_BGR_888);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)_image.data, _width, _height, RK_FORMAT_RGB_888);
    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR == ret)
    {
        ret = imresize(src, dst);
        if (IM_STATUS_SUCCESS == ret)
        {
            _inputs[0].buf = _image.data;
            cv::imwrite("Image.png", _image);
            return true;
        }
        else
        {
            fprintf(stderr, "resize error %d %s\n", ret, imStrError((IM_STATUS)ret));
        }
    }
    else
    {
        fprintf(stderr, "rga check error! %s\n", imStrError((IM_STATUS)ret));
    }
    return false;
#endif
}

void YOLOv5::PostProcess()
{
    GenerateProposals((int8_t*)_outputs[0].buf, anchor0, strides[0], _output_attrs[0].zp, _output_attrs[0].scale);
    GenerateProposals((int8_t*)_outputs[1].buf, anchor1, strides[1], _output_attrs[1].zp, _output_attrs[1].scale);
    GenerateProposals((int8_t*)_outputs[2].buf, anchor2, strides[2], _output_attrs[2].zp, _output_attrs[2].scale);
  
    // Sort by probability
    if (2 <= _proposals.size())
    {
        std::sort(_proposals.begin(), _proposals.end());
    }
    std::vector<int> picked = nmsSortedBoxes();
    printf("proposals:%ld picked:%ld\n", _proposals.size(), picked.size());
    
    _objects.resize(picked.size());
    for (size_t i = 0; i < picked.size(); i++)
    {
        _objects[i] = _proposals[picked[i]];
    }
    for (int i = 0; i < _objects.size(); i++)
    {
        printf("x:%f y:%f w:%f h:%f %f %s\n", 
            _objects[i].box.x, _objects[i].box.y, _objects[i].box.width, _objects[i].box.height,
            _objects[i].prob, coco_80_labels[_objects[i].id]);
    }
}

inline static int clamp(float val, int min, int max) 
{
    return val > min ? (val < max ? val : max) : min; 
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

void YOLOv5::generateProposals(Result* results, const std::vector<GridAndStride> grid, int zp, float scale)
{
    for (size_t i = 0; i < grid.size(); i++)
    {
        //float stride = grid[i].stride;
        float x = deqnt_affine_to_f32(results[i].x, zp, scale) / _scale_x;
        float y = deqnt_affine_to_f32(results[i].y, zp, scale) / _scale_y;
        float w = deqnt_affine_to_f32(results[i].w, zp, scale) / _scale_x;
        float h = deqnt_affine_to_f32(results[i].h, zp, scale) / _scale_y;
        float box_objectness = deqnt_affine_to_f32(results[i].box_prob, zp, scale);
        
        if (_box_conf_threshold < box_objectness)
        {
            OBJECT object = {{x, y, w, h}, box_objectness};

            // Choose class
            int max_class_id = 0;
            int8_t max_class_prob = results[i].class_score[0]; 
            for (int class_id = 0; class_id < OBJ_CLASS_NUM; class_id++)
            {
                int8_t prob = results[i].class_score[class_id];
                if (max_class_prob < prob)
                {
                    max_class_id = class_id;
                    max_class_prob = prob;
                }
            }
            object.id = max_class_id;
            object.prob = deqnt_affine_to_f32(max_class_prob, zp, scale) * box_objectness;
            _proposals.push_back(object);
        }
    }
}

std::vector<int> YOLOv5::nmsSortedBoxes()
{
    std::vector<int> picked;
    std::vector<float> areas(_proposals.size());
    for (size_t i = 0; i < _proposals.size(); i++)
    {
        areas[i] = _proposals[i].box.area();
    }
    
    for (size_t i = 0; i < _proposals.size(); i++)
    {
        const OBJECT& a = _proposals[i];

        int keep = 1;
        for (size_t j = 0; j < picked.size(); j++)
        {
            const OBJECT& b = _proposals[picked[j]];

            // intersection over union
            float inter_area = (a.box & b.box).area();
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if ((inter_area / union_area) < _nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back((int)i);
    }
    return picked;
}



void YOLOv5::GenerateProposals(int8_t *input, int *anchor, int stride, int zp, float scale)
{
    int grid_w = _width / stride;
    int grid_h = _height / stride;
    int grid_len = grid_h * grid_w;
    //int8_t thres_i8 = qnt_f32_to_affine(_box_conf_threshold, zp, scale);
    printf("w:%d h:%d thres:%f\n", grid_w, grid_h, _box_conf_threshold);
    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                float box_confidence = deqnt_affine_to_f32(input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j], zp, scale);
                //printf("conf:%d\n", box_confidence);
                if (_box_conf_threshold <= box_confidence)
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

                    float maxClassProbs = deqnt_affine_to_f32(in_ptr[5 * grid_len], zp, scale);
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                    {
                        float prob = deqnt_affine_to_f32(in_ptr[(5 + k) * grid_len], zp, scale);
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    //if (thres_i8 <= maxClassProbs)
                    {
                        OBJECT proposal;
                        proposal.id = maxClassId;
                        proposal.prob = maxClassProbs * box_confidence;
                        proposal.box = {box_x, box_y, box_w, box_h};
                        _proposals.push_back(proposal);
                    }
                }
            }
        }
    }
}
