#include <opencv2/core.hpp>
#include "Inference.hpp"

class YOLOX: RKNN
{
public:
    YOLOX();

    ~YOLOX();

    bool Initialize(const char* model_filepath, float nms_threshold, float box_threshold, rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2);

    cv::Mat Infer(cv::Mat& image);

protected:
    float _nms_threshold;
    float _box_conf_threshold;
    int _channel;
    int _width;
    int _height;
    rknn_input* _inputs;
    rknn_output* _outputs;
    cv::Mat _image;
};
