#include <sys/time.h>
#include "YOLOv8.hpp"

#define PROTO_CHANNEL 32
#define PROTO_HEIGHT 160
#define PROTO_WEIGHT 160

static double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

YOLOv8::YOLOv8(): RKNN(), _channel(3), _inputs(nullptr), _outputs(nullptr), _scale(1), _offset_x(0), _offset_y(0)
{

}

YOLOv8::~YOLOv8()
{
    if (_inputs != nullptr)
        delete[] _inputs;
    if (_outputs != nullptr)
        delete[] _outputs;
    _inputs = nullptr;
    _outputs = nullptr;
}

bool YOLOv8::Initialize(const char* model_path, float box_threshold, float nms_threshold, rknn_core_mask core_mask)
{
    if (RKNN::Initialize(model_path, core_mask))
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
        _buffer.create(_height, _width, CV_8UC3);

        printf("%d outputs %s\n", _io_num.n_output, get_type_string(_output_attrs[0].type));
        _outputs = new rknn_output[_io_num.n_output];
        for (int i = 0; i < _io_num.n_output; i++)
        {
            memset(&_outputs[i], 0, sizeof(rknn_output));
            _outputs[i].index = i;
            _outputs[i].want_float = (_output_attrs[i].type == RKNN_TENSOR_FLOAT16);
        }
        return true;
    }
    return false;
}

std::vector<OBJECT> YOLOv8::Detect(cv::Mat image)
{   
    timeval start, stop;
    preProcess(image);
        
    gettimeofday(&start, NULL);

    Inference(_inputs, _outputs);

    gettimeofday(&stop, NULL);
    printf("once run use %f ms\n", (__get_us(stop) - __get_us(start)) / 1000);

    return postProcess();
}

void YOLOv8::preProcess(cv::Mat image)
{
    _scale = image.rows > image.cols? (float)_height / image.rows: (float)_width / image.cols;

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(), _scale, _scale);

    _offset_x = (_width - resized_image.cols) / 2;
    _offset_y = (_height - resized_image.rows) / 2;

    printf("W:%d H:%d Scale:%f top:%d left:%d\n", image.cols, image.rows, _scale, _offset_y, _offset_x);
    // Cast resized image
    cv::copyMakeBorder(resized_image, _buffer, _offset_y, _height - (_offset_y + resized_image.rows), _offset_x, _width - (_offset_x + resized_image.cols), cv::BORDER_CONSTANT, cv::Scalar(128, 0, 0));
    cv::imwrite("letterbox.png", _buffer);
    
    // Set input blob
    _inputs[0].buf = _buffer.data;
}
    
std::vector<OBJECT> YOLOv8::postProcess()
{
    std::vector<OBJECT> objects;
    int dfl_len = _output_attrs[0].dims[1] / 4;

    _proposals.clear();
    for (int i = 0; i < _io_num.n_output; i += 3)
    {
        int grid_h = _output_attrs[i].dims[2];
        int grid_w = _output_attrs[i].dims[3];
        int stride = _height / grid_h;
        printf("outputs[%d] h:%d w:%d stride:%d %s\n", i, grid_h, grid_w, stride, get_type_string(_output_attrs[i].type));
        if (_output_attrs[i].type == RKNN_TENSOR_FLOAT16)
            process_float(i, grid_h, grid_w, stride, dfl_len);
        else
            process_int(i, grid_h, grid_w, stride, dfl_len);
    }

    std::sort(_proposals.begin(), _proposals.end());
    std::vector<int> picked = nms(_proposals);
    printf("%ld proposals %ld detected\n", _proposals.size(), objects.size());
    for (int i = 0; i < picked.size(); i++)
    {
        OBJECT& object = _proposals[picked[i]];

        object.box.x -= _offset_x;
        object.box.y -= _offset_y;
        object.box.x /= _scale;
        object.box.y /= _scale;
        object.box.width /= _scale;
        object.box.height /= _scale;

        objects.push_back(object);
        printf("%s x:%f y:%f width:%f height:%f\n", categories[object.id], object.box.x, object.box.y, object.box.width, object.box.height);
    }
 
    return objects;
}

std::vector<int> YOLOv8::nms(const std::vector<OBJECT>& objects)
{
    std::vector<int>picked;

    const size_t n = objects.size();

    std::vector<float> areas(n);
    for (size_t i = 0; i < n; i++)
    {
        areas[i] = objects[i].box.area();
    }

    for (size_t i = 0; i < n; i++)
    {
        const OBJECT& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const OBJECT& b = objects[picked[j]];

            // intersection over union
            float inter_area = (a.box & b.box).area();
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (_nms_threshold < inter_area / union_area)
                keep = 0;
        }

        if (keep)
            picked.push_back((int)i);
    }
    return picked;
}

static void compute_dfl(float *tensor, int dfl_len, float *box)
{
    for (int b = 0; b < 4; b++)
    {
        float exp_t[dfl_len];
        float exp_sum = 0;
        float acc_sum = 0;
        for (int i = 0; i < dfl_len; i++)
        {
            exp_t[i] = exp(tensor[i + b * dfl_len]);
            exp_sum += exp_t[i];
        }

        for (int i = 0; i < dfl_len; i++)
        {
            acc_sum += exp_t[i] / exp_sum * i;
        }
        box[b] = acc_sum;
    }
}

void YOLOv8::process_float(int index, int grid_h, int grid_w, int stride, int dfl_len)
{
    int grid_len = grid_h * grid_w;
    float *box_tensor = (float *)_outputs[index].buf;
    float *score_tensor = (float *)_outputs[index + 1].buf;
    float *score_sum_tensor = (float *)_outputs[index + 2].buf;
    //float *seg_tensor = (float *)_outputs[index + 3].buf;

    for (int y = 0; y < grid_h; y++)
    {
        for (int x = 0; x < grid_w; x++)
        {
            int offset = y * grid_w + x;
            int max_class_id = 0;
            float max_score = 0;
            int offset_seg = y * grid_w + x;
            //float *in_ptr_seg = seg_tensor + offset_seg;

            // for quick filtering through "score sum"
            if (score_sum_tensor != nullptr)
            {
                //printf("%d, %d, %f", y, x, score_sum_tensor[offset]);
                if (score_sum_tensor[offset] < _box_conf_threshold)
                {
                    //puts("");
                    continue;
                }
            }

            for (int c = 0; c < OBJ_CLASS_NUM; c++)
            {
                if ((score_tensor[offset] > _box_conf_threshold) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }
            
            // compute box
            if (max_score > _box_conf_threshold)
            {
                /* segmentation data
                for (int k = 0; k < PROTO_CHANNEL; k++)
                {
                    float seg_element_f32 = in_ptr_seg[(k)*grid_len];
                    segments.push_back(seg_element_f32);
                }
                */

                offset = y * grid_w + x;
                float box[4];
                float before_dfl[dfl_len * 4];
                for (int k = 0; k < dfl_len * 4; k++)
                {
                    before_dfl[k] = box_tensor[offset];
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float left, top, right, bottom, w, h;
                left = (-box[0] + x + 0.5) * stride;
                top = (-box[1] + y + 0.5) * stride;
                right = (box[2] + x + 0.5) * stride;
                bottom = (box[3] + y + 0.5) * stride;
                w = right - left;
                h = bottom - top;
                //printf("%s, %f, %f, %f, %f, %f\n", categories[max_class_id], max_score, left, top, w, h);

                OBJECT object = {{left, top, w, h}, max_score, max_class_id};
                _proposals.push_back(object);
            }
        }
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

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

void YOLOv8::process_int(int index, int grid_h, int grid_w, int stride, int dfl_len)
{
    int grid_len = grid_h * grid_w;
    int8_t *box_tensor = (int8_t *)_outputs[index].buf;
    int8_t *score_tensor = (int8_t *)_outputs[index + 1].buf;
    int8_t *score_sum_tensor = (int8_t *)_outputs[index + 2].buf;
    int8_t *seg_tensor = (int8_t *)_outputs[index + 3].buf;

    // dequatize parameters
    int32_t box_zp = _output_attrs[index].zp;
    float box_scale = _output_attrs[index].scale;
    int32_t score_zp = _output_attrs[index + 1].zp;
    float score_scale = _output_attrs[index + 1].scale;
    int32_t score_sum_zp = _output_attrs[index + 2].zp;
    float score_sum_scale = _output_attrs[index + 2].scale;
    int32_t seg_zp = _output_attrs[index + 2].zp;
    float seg_scale = _output_attrs[index + 2].scale;

    int8_t score_threshold = qnt_f32_to_affine(_box_conf_threshold, score_zp, score_scale);
    int8_t score_sum_threshold= qnt_f32_to_affine(_box_conf_threshold, score_sum_zp, score_sum_scale);

    for (int y = 0; y < grid_h; y++)
    {
        for (int x = 0; x < grid_w; x++)
        {
            int offset = y * grid_w + x;
            int max_class_id = 0;
            int8_t max_score = -score_zp;
            int offset_seg = y * grid_w + x;
            int8_t *in_ptr_seg = seg_tensor + offset_seg;

            // for quick filtering through "score sum"
            if (score_sum_tensor != nullptr)
            {
                if (score_sum_tensor[offset] < score_sum_threshold)
                {
                    continue;
                }
            }

            for (int c = 0; c < OBJ_CLASS_NUM; c++)
            {
                if ((score_tensor[offset] > score_threshold) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }
            
            // compute box
            if (max_score > _box_conf_threshold)
            {
                /* segmentation data
                for (int k = 0; k < PROTO_CHANNEL; k++)
                {
                    float seg_element_f32 = deqnt_affine_to_f32(in_ptr_seg[(k)*grid_len], seg_zp, seg_scale);
                    segments.push_back(seg_element_f32);
                }
                */

                offset = y * grid_w + x;
                float box[4];
                float before_dfl[dfl_len * 4];
                for (int k = 0; k < dfl_len * 4; k++)
                {
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float left, top, right, bottom, w, h;
                left = (-box[0] + x + 0.5) * stride;
                top = (-box[1] + y + 0.5) * stride;
                right = (box[2] + x + 0.5) * stride;
                bottom = (box[3] + y + 0.5) * stride;
                w = right - left;
                h = bottom - top;
                
                OBJECT object = {{left, top, w, h}, deqnt_affine_to_f32(max_score, score_zp, score_scale), max_class_id};
                _proposals.push_back(object);
            }
        }
    }
}