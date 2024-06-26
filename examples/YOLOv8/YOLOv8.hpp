#ifndef YOLOv8_INCLUDED

#include <opencv2/opencv.hpp>
#include "Inference.hpp"

#define OBJ_CLASS_NUM 80
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25


static const char *categories[OBJ_CLASS_NUM]
{
#include "coco_80_labels.h"
};

typedef struct _OBJECT
{
    cv::Rect_<float> box;
    float prob;    
    int id;
    float box_confidence;
    // Comparison for sort
    bool operator<(const _OBJECT& right) const { return prob > right.prob; }
    void Draw(cv::Mat& image)
    {
        char text[256];
        cv::rectangle(image, box, cv::Scalar(255, 255, 255), 2);
        sprintf(text, "%s %.1f%%", categories[id], prob * 100);
        cv::putText(image, text, cv::Point(box.x, box.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2, 8, 0);
    }
} OBJECT;

class YOLOv8: public RKNN
{
public:
    YOLOv8();
    ~YOLOv8();

    bool Initialize(const char* model_path, float box_conf_threshold = BOX_THRESH, float nms_threshold = NMS_THRESH, rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2);

    std::vector<OBJECT> Detect(cv::Mat image);

protected:
    void preProcess(cv::Mat image);
    std::vector<OBJECT> postProcess();
    void process_float(int index, int h, int w, int stride, int dfl_len);
    void process_int(int index, int h, int w, int stride, int dfl_len);
    std::vector<int> nms(const std::vector<OBJECT>& objects);

protected:
    int _channel;
    int _width;
    int _height;
    int _image_width;
    int _image_height;
    bool is_quant;
    float _nms_threshold;
    float _box_conf_threshold;
    float _scale;
    int _offset_x;
    int _offset_y;
    rknn_input* _inputs;
    rknn_output* _outputs;
    cv::Mat _buffer;
    std::vector<OBJECT> _proposals;
};

#define YOLOv8_INCLUDED
#endif