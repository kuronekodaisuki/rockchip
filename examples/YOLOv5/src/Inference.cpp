#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/time.h>

#include "Inference.hpp"

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
  std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
  for (int i = 1; i < attr->n_dims; ++i)
  {
    shape_str += ", " + std::to_string(attr->dims[i]);
  }

  printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
         "type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
         attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
  unsigned char *data;
  int ret;

  data = NULL;

  if (NULL == fp)
  {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0)
  {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = new unsigned char[sz];
  if (data == NULL)
  {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
  FILE *fp;
  unsigned char *data;

  fp = fopen(filename, "rb");
  if (NULL == fp)
  {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

RKNN::RKNN()
{

}

RKNN::~RKNN()
{
    if (_ctx != 0)
      rknn_destroy(_ctx);

    if (_model_data != nullptr)
        delete[] _model_data;

    if (_input_attrs != nullptr)
        delete[] _input_attrs;
    if (_output_attrs != nullptr)
        delete[] _output_attrs;
    _model_data = nullptr;
    _input_attrs = _output_attrs = nullptr;
}

bool RKNN::Initialize(const char* model_filepath, rknn_core_mask core_mask)
{
    int ret = 0;
    int model_data_size = 0;
    _model_data = load_model(model_filepath, &model_data_size);
    ret = rknn_init(&_ctx, _model_data, model_data_size, 0, NULL);

    ret = rknn_query(_ctx, RKNN_QUERY_IN_OUT_NUM, &_io_num, sizeof(_io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return false;
    }
    printf("model input num: %d, output num: %d\n", _io_num.n_input, _io_num.n_output);

    // Set the input parameters
    _input_attrs = new rknn_tensor_attr[_io_num.n_input];
    for (int i = 0; i < _io_num.n_input; i++)
    {
        _input_attrs[i].index = i;
        ret = rknn_query(_ctx, RKNN_QUERY_INPUT_ATTR, &(_input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return false;
        }
        dump_tensor_attr(&(_input_attrs[i]));
     }

    // Set the output parameters
    _output_attrs = new rknn_tensor_attr[_io_num.n_output];
    for (int i = 0; i < _io_num.n_output; i++)
    {
        _output_attrs[i].index = i;
        ret = rknn_query(_ctx, RKNN_QUERY_OUTPUT_ATTR, &(_output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            return false;
        }
        dump_tensor_attr(&(_output_attrs[i]));
    }
    return true;
}

