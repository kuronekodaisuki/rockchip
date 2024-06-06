//
//
//
#ifndef YOLOX_INCLUDED
#include <vector>
#include <opencv2/core.hpp>
#include "Inference.hpp"

//#define OBJ_NAME_MAX_SIZE 16
//#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

typedef struct _OBJECT
{
    cv::Rect_<float> box;
    float prob;    
    int id;
    bool operator<(const _OBJECT& right) const { return prob > right.prob; }
} OBJECT;

typedef struct
{
    int stride;
    int x;
    int y;
} GridAndStride;

typedef struct
{
    int8_t x;
    int8_t y;
    int8_t w;
    int8_t h;
    int8_t box_prob;
    int8_t class_score[OBJ_CLASS_NUM]; 
} Result;

class YOLOX: RKNN
{
public:
    YOLOX();

    ~YOLOX();

    bool Initialize(const char* model_filepath, float nms_threshold = NMS_THRESH, float box_threshold = BOX_THRESH, rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2);

    cv::Mat Infer(cv::Mat& image);

private:
    void PostProcess();
    void GenerateProposals(int8_t *input, int *anchor, int stride, int zp, float scale);
    void generateProposals(Result* input, const std::vector<GridAndStride> grid, int zp, float scale);
    std::vector<int> nmsSortedBoxes();

protected:
    float _nms_threshold;
    float _box_conf_threshold;
    int _channel;
    int _width;
    int _height;
    float _scale_x;
    float _scale_y;
    rknn_input* _inputs;
    rknn_output* _outputs;
    cv::Mat _image;
    std::vector<OBJECT> _proposals;
    std::vector<OBJECT> _objects;
    std::vector<GridAndStride> _grids[3];
};
#define YOLOX_INCLUDED
#endif