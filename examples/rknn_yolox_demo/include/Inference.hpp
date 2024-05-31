//
// Inference.h -- RKNN Inference class definition
//
#ifndef INFERENCE_INCLUDED
#include "rknn_api.h"

class RKNN
{
public:
    RKNN();
    bool Initialize(const char* model_filepath, rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2);

    virtual bool Do();
    virtual bool PreProcess();
    virtual bool PostProcess();

    ~RKNN();
protected:
    unsigned char *_model_data = nullptr;
    rknn_context _ctx;
    rknn_input_output_num _io_num;
    rknn_tensor_attr *_input_attrs = nullptr;
    rknn_tensor_attr *_output_attrs = nullptr;
};

#define INFERENCE_INCLUDED
#endif


