#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "../3rdparty/rga/include/im2d.h"
#include "../3rdparty/rga/include/drmrga.h"
#include "image_utils.hpp"
#include "file_utils.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_THREAD_LOCALS
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG

#define STB_IMAGE_WRITE_IMPLEMENTATION




static const char* filter_image_names[] = {
    "jpg",
    "jpeg",
    "JPG",
    "JPEG",
    "png",
    "PNG",
    "data",
    NULL
};
static int crop_and_scale_image_c(int channel, unsigned char *src, int src_width, int src_height,
                                    int crop_x, int crop_y, int crop_width, int crop_height,
                                    unsigned char *dst, int dst_width, int dst_height,
                                    int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height) {
    if (dst == NULL) {
        printf("dst buffer is null\n");
        return -1;
    }

    float x_ratio = (float)crop_width / (float)dst_box_width;
    float y_ratio = (float)crop_height / (float)dst_box_height;

    // printf("src_width=%d src_height=%d crop_x=%d crop_y=%d crop_width=%d crop_height=%d\n",
    //     src_width, src_height, crop_x, crop_y, crop_width, crop_height);
    // printf("dst_width=%d dst_height=%d dst_box_x=%d dst_box_y=%d dst_box_width=%d dst_box_height=%d\n",
    //     dst_width, dst_height, dst_box_x, dst_box_y, dst_box_width, dst_box_height);
    // printf("channel=%d x_ratio=%f y_ratio=%f\n", channel, x_ratio, y_ratio);

    // 从原图指定区域取数据，双线性缩放到目标指定区域
    for (int dst_y = dst_box_y; dst_y < dst_box_y + dst_box_height; dst_y++) {
        for (int dst_x = dst_box_x; dst_x < dst_box_x + dst_box_width; dst_x++) {
            int dst_x_offset = dst_x - dst_box_x;
            int dst_y_offset = dst_y - dst_box_y;

            int src_x = (int)(dst_x_offset * x_ratio) + crop_x;
            int src_y = (int)(dst_y_offset * y_ratio) + crop_y;

            float x_diff = (dst_x_offset * x_ratio) - (src_x - crop_x);
            float y_diff = (dst_y_offset * y_ratio) - (src_y - crop_y);

            int index1 = src_y * src_width * channel + src_x * channel;
            int index2 = index1 + src_width * channel;    // down
            if (src_y == src_height - 1) {
                // 如果到图像最下边缘，变成选择上面的像素
                index2 = index1 - src_width * channel;
            }
            int index3 = index1 + 1 * channel;            // right
            int index4 = index2 + 1 * channel;            // down right
            if (src_x == src_width - 1) {
                // 如果到图像最右边缘，变成选择左边的像素
                index3 = index1 - 1 * channel;
                index4 = index2 - 1 * channel;
            }

            // printf("dst_x=%d dst_y=%d dst_x_offset=%d dst_y_offset=%d src_x=%d src_y=%d x_diff=%f y_diff=%f src index=%d %d %d %d\n",
            //     dst_x, dst_y, dst_x_offset, dst_y_offset,
            //     src_x, src_y, x_diff, y_diff,
            //     index1, index2, index3, index4);

            for (int c = 0; c < channel; c++) {
                unsigned char A = src[index1+c];
                unsigned char B = src[index3+c];
                unsigned char C = src[index2+c];
                unsigned char D = src[index4+c];

                unsigned char pixel = (unsigned char)(
                    A * (1 - x_diff) * (1 - y_diff) +
                    B * x_diff * (1 - y_diff) +
                    C * y_diff * (1 - x_diff) +
                    D * x_diff * y_diff
                );

                dst[(dst_y * dst_width  + dst_x) * channel + c] = pixel;
            }
        }
    }

    return 0;
}

static int crop_and_scale_image_yuv420sp(unsigned char *src, int src_width, int src_height,
                                    int crop_x, int crop_y, int crop_width, int crop_height,
                                    unsigned char *dst, int dst_width, int dst_height,
                                    int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height) {

    unsigned char* src_y = src;
    unsigned char* src_uv = src + src_width * src_height;

    unsigned char* dst_y = dst;
    unsigned char* dst_uv = dst + dst_width * dst_height;

    crop_and_scale_image_c(1, src_y, src_width, src_height, crop_x, crop_y, crop_width, crop_height,
        dst_y, dst_width, dst_height, dst_box_x, dst_box_y, dst_box_width, dst_box_height);
    
    crop_and_scale_image_c(2, src_uv, src_width / 2, src_height / 2, crop_x / 2, crop_y / 2, crop_width / 2, crop_height / 2,
        dst_uv, dst_width / 2, dst_height / 2, dst_box_x, dst_box_y, dst_box_width, dst_box_height);

    return 0;
}

static int convert_image_cpu(image_buffer_t *src, image_buffer_t *dst, image_rect_t *src_box, image_rect_t *dst_box, char color) {
    int ret;
    if (dst->virt_addr == NULL) {
        return -1;
    }
    if (src->virt_addr == NULL) {
        return -1;
    }
    if (src->format != dst->format) {
        return -1;
    }

    int src_box_x = 0;
    int src_box_y = 0;
    int src_box_w = src->width;
    int src_box_h = src->height;
    if (src_box != NULL) {
        src_box_x = src_box->left;
        src_box_y = src_box->top;
        src_box_w = src_box->right - src_box->left + 1;
        src_box_h = src_box->bottom - src_box->top + 1;
    }
    int dst_box_x = 0;
    int dst_box_y = 0;
    int dst_box_w = dst->width;
    int dst_box_h = dst->height;
    if (dst_box != NULL) {
        dst_box_x = dst_box->left;
        dst_box_y = dst_box->top;
        dst_box_w = dst_box->right - dst_box->left + 1;
        dst_box_h = dst_box->bottom - dst_box->top + 1;
    }

    // fill pad color
    if (dst_box_w != dst->width || dst_box_h != dst->height) {
        int dst_size = get_image_size(dst);
        memset(dst->virt_addr, color, dst_size);
    }

    int need_release_dst_buffer = 0;
    int reti = 0;
    if (src->format == IMAGE_FORMAT_RGB888) {
        reti = crop_and_scale_image_c(3, src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else if (src->format == IMAGE_FORMAT_RGBA8888) {
        reti = crop_and_scale_image_c(4, src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else if (src->format == IMAGE_FORMAT_GRAY8) {
        reti = crop_and_scale_image_c(1, src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else if (src->format == IMAGE_FORMAT_YUV420SP_NV12 || src->format == IMAGE_FORMAT_YUV420SP_NV12) {
        reti = crop_and_scale_image_yuv420sp(src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else {
        printf("no support format %d\n", src->format);
    }
    if (reti != 0) {
        printf("convert_image_cpu fail %d\n", reti);
        return -1;
    }
    printf("finish\n");
    return 0;
}

static int get_rga_fmt(image_format_t fmt) {
    switch (fmt)
    {
    case IMAGE_FORMAT_RGB888:
        return RK_FORMAT_RGB_888;
    case IMAGE_FORMAT_RGBA8888:
        return RK_FORMAT_RGBA_8888;
    case IMAGE_FORMAT_YUV420SP_NV12:
        return RK_FORMAT_YCbCr_420_SP;
    case IMAGE_FORMAT_YUV420SP_NV21:
        return RK_FORMAT_YCrCb_420_SP;
    default:
        return -1;
    }
}

int get_image_size(image_buffer_t* image)
{
    if (image == NULL) {
        return 0;
    }
    switch (image->format)
    {
    case IMAGE_FORMAT_GRAY8:
        return image->width * image->height;
    case IMAGE_FORMAT_RGB888:
        return image->width * image->height * 3;    
    case IMAGE_FORMAT_RGBA8888:
        return image->width * image->height * 4;
    case IMAGE_FORMAT_YUV420SP_NV12:
    case IMAGE_FORMAT_YUV420SP_NV21:
        return image->width * image->height * 3 / 2;
    default:
        break;
    }
}

static int convert_image_rga(image_buffer_t* src_img, image_buffer_t* dst_img, image_rect_t* src_box, image_rect_t* dst_box, char color)
{
    int ret = 0;

    int srcWidth = src_img->width;
    int srcHeight = src_img->height;
    void *src = src_img->virt_addr;
    int src_fd = src_img->fd;
    void *src_phy = NULL;
    int srcFmt = get_rga_fmt(src_img->format);

    int dstWidth = dst_img->width;
    int dstHeight = dst_img->height;
    void *dst = dst_img->virt_addr;
    int dst_fd = dst_img->fd;
    void *dst_phy = NULL;
    int dstFmt = get_rga_fmt(dst_img->format);

    int rotate = 0;

    int use_handle = 0;
#if defined(LIBRGA_IM2D_HANDLE)
    use_handle = 1;
#endif

    // printf("src width=%d height=%d fmt=0x%x virAddr=0x%p fd=%d\n",
    //     srcWidth, srcHeight, srcFmt, src, src_fd);
    // printf("dst width=%d height=%d fmt=0x%x virAddr=0x%p fd=%d\n",
    //     dstWidth, dstHeight, dstFmt, dst, dst_fd);
    // printf("rotate=%d\n", rotate);

    int usage = 0;
    IM_STATUS ret_rga = IM_STATUS_NOERROR;

    // set rga usage
    usage |= rotate;

    // set rga rect
    im_rect srect;
    im_rect drect;
    im_rect prect;
    memset(&prect, 0, sizeof(im_rect));

    if (src_box != NULL) {
        srect.x = src_box->left;
        srect.y = src_box->top;
        srect.width = src_box->right - src_box->left + 1;
        srect.height = src_box->bottom - src_box->top + 1;
    } else {
        srect.x = 0;
        srect.y = 0;
        srect.width = srcWidth;
        srect.height = srcHeight;
    }

    if (dst_box != NULL) {
        drect.x = dst_box->left;
        drect.y = dst_box->top;
        drect.width = dst_box->right - dst_box->left + 1;
        drect.height = dst_box->bottom - dst_box->top + 1;
    } else {
        drect.x = 0;
        drect.y = 0;
        drect.width = dstWidth;
        drect.height = dstHeight;
    }

    // set rga buffer
    rga_buffer_t rga_buf_src;
    rga_buffer_t rga_buf_dst;
    rga_buffer_t pat;
    rga_buffer_handle_t rga_handle_src = 0;
    rga_buffer_handle_t rga_handle_dst = 0;
    memset(&pat, 0, sizeof(rga_buffer_t));

    im_handle_param_t in_param;
    in_param.width = srcWidth;
    in_param.height = srcHeight;
    in_param.format = srcFmt;

    im_handle_param_t dst_param;
    dst_param.width = dstWidth;
    dst_param.height = dstHeight;
    dst_param.format = dstFmt;

    if (use_handle) {
        if (src_phy != NULL) {
            rga_handle_src = importbuffer_physicaladdr((uint64_t)src_phy, &in_param);
        } else if (src_fd > 0) {
            rga_handle_src = importbuffer_fd(src_fd, &in_param);
        } else {
            rga_handle_src = importbuffer_virtualaddr(src, &in_param);
        }
        if (rga_handle_src <= 0) {
            printf("src handle error %d\n", rga_handle_src);
            ret = -1;
            return ret;
        }
        rga_buf_src = wrapbuffer_handle(rga_handle_src, srcWidth, srcHeight, srcFmt, srcWidth, srcHeight);
    } else {
        if (src_phy != NULL) {
            rga_buf_src = wrapbuffer_physicaladdr(src_phy, srcWidth, srcHeight, srcFmt, srcWidth, srcHeight);
        } else if (src_fd > 0) {
            rga_buf_src = wrapbuffer_fd(src_fd, srcWidth, srcHeight, srcFmt, srcWidth, srcHeight);
        } else {
            rga_buf_src = wrapbuffer_virtualaddr(src, srcWidth, srcHeight, srcFmt, srcWidth, srcHeight);
        }
    }

    if (use_handle) {
        if (dst_phy != NULL) {
            rga_handle_dst = importbuffer_physicaladdr((uint64_t)dst_phy, &dst_param);
        } else if (dst_fd > 0) {
            rga_handle_dst = importbuffer_fd(dst_fd, &dst_param);
        } else {
            rga_handle_dst = importbuffer_virtualaddr(dst, &dst_param);
        }
        if (rga_handle_dst <= 0) {
            printf("dst handle error %d\n", rga_handle_dst);
            ret = -1;
            return ret;
        }
        rga_buf_dst = wrapbuffer_handle(rga_handle_dst, dstWidth, dstHeight, dstFmt, dstWidth, dstHeight);
    } else {
        if (dst_phy != NULL) {
            rga_buf_dst = wrapbuffer_physicaladdr(dst_phy, dstWidth, dstHeight, dstFmt, dstWidth, dstHeight);
        } else if (dst_fd > 0) {
            rga_buf_dst = wrapbuffer_fd(dst_fd, dstWidth, dstHeight, dstFmt, dstWidth, dstHeight);
        } else {
            rga_buf_dst = wrapbuffer_virtualaddr(dst, dstWidth, dstHeight, dstFmt, dstWidth, dstHeight);
        }
    }

    if (drect.width != dstWidth || drect.height != dstHeight) {
        im_rect dst_whole_rect = {0, 0, dstWidth, dstHeight};
        int imcolor;
        char* p_imcolor = (char*)&imcolor;
        p_imcolor[0] = color;
        p_imcolor[1] = color;
        p_imcolor[2] = color;
        p_imcolor[3] = color;
        printf("fill dst image (x y w h)=(%d %d %d %d) with color=0x%x\n",
            dst_whole_rect.x, dst_whole_rect.y, dst_whole_rect.width, dst_whole_rect.height, imcolor);
        ret_rga = imfill(rga_buf_dst, dst_whole_rect, imcolor);
        if (ret_rga <= 0) {
            if (dst != NULL) {
                size_t dst_size = get_image_size(dst_img);
                memset(dst, color, dst_size);
            } else {
                printf("Warning: Can not fill color on target image\n");
            }
        }
    }

    // rga process
    ret_rga = improcess(rga_buf_src, rga_buf_dst, pat, srect, drect, prect, usage);
    if (ret_rga <= 0) {
        printf("Error on improcess STATUS=%d\n", ret_rga);
        printf("RGA error message: %s\n", imStrError((IM_STATUS)ret_rga));
        ret = -1;
    }

err:
    if (rga_handle_src > 0) {
        releasebuffer_handle(rga_handle_src);
    }

    if (rga_handle_dst > 0) {
        releasebuffer_handle(rga_handle_dst);
    }

    // printf("finish\n");
    return ret;
}

int convert_image(image_buffer_t* src_img, image_buffer_t* dst_img, image_rect_t* src_box, image_rect_t* dst_box, char color)
{
    int ret;
 
    printf("src width=%d height=%d fmt=0x%x virAddr=0x%p fd=%d\n",
        src_img->width, src_img->height, src_img->format, src_img->virt_addr, src_img->fd);
    printf("dst width=%d height=%d fmt=0x%x virAddr=0x%p fd=%d\n",
        dst_img->width, dst_img->height, dst_img->format, dst_img->virt_addr, dst_img->fd);
    if (src_box != NULL) {
        printf("src_box=(%d %d %d %d)\n", src_box->left, src_box->top, src_box->right, src_box->bottom);
    }
    if (dst_box != NULL) {
        printf("dst_box=(%d %d %d %d)\n", dst_box->left, dst_box->top, dst_box->right, dst_box->bottom);
    }
    printf("color=0x%x\n", color);

    ret = convert_image_rga(src_img, dst_img, src_box, dst_box, color);
    if (ret != 0) {
        printf("try convert image use cpu\n");
        ret = convert_image_cpu(src_img, dst_img, src_box, dst_box, color);
    }
    return ret;
}

int convert_image_with_letterbox(image_buffer_t* src_image, image_buffer_t* dst_image, letterbox_t* letterbox, char color)
{
    int ret = 0;
    int allow_slight_change = 1;
    int src_w = src_image->width;
    int src_h = src_image->height;
    int dst_w = dst_image->width;
    int dst_h = dst_image->height;
    int resize_w = dst_w;
    int resize_h = dst_h;

    int padding_w = 0;
    int padding_h = 0;

    int _left_offset = 0;
    int _top_offset = 0;
    float scale = 1.0;

    image_rect_t src_box;
    src_box.left = 0;
    src_box.top = 0;
    src_box.right = src_image->width - 1;
    src_box.bottom = src_image->height - 1;

    image_rect_t dst_box;
    dst_box.left = 0;
    dst_box.top = 0;
    dst_box.right = dst_image->width - 1;
    dst_box.bottom = dst_image->height - 1;

    float _scale_w = (float)dst_w / src_w;
    float _scale_h = (float)dst_h / src_h;
    if(_scale_w < _scale_h) {
        scale = _scale_w;
        resize_h = (int) src_h*scale;
    } else {
        scale = _scale_h;
        resize_w = (int) src_w*scale;
    }
    // slight change image size for align
    if (allow_slight_change == 1 && (resize_w % 4 != 0)) {
        resize_w -= resize_w % 4;
    }
    if (allow_slight_change == 1 && (resize_h % 2 != 0)) {
        resize_h -= resize_h % 2;
    }
    // padding
    padding_h = dst_h - resize_h;
    padding_w = dst_w - resize_w;
    // center
    if (_scale_w < _scale_h) {
        dst_box.top = padding_h / 2;
        if (dst_box.top % 2 != 0) {
            dst_box.top -= dst_box.top % 2;
            if (dst_box.top < 0) {
                dst_box.top = 0;
            }
        }
        dst_box.bottom = dst_box.top + resize_h - 1;
        _top_offset = dst_box.top;
    } else {
        dst_box.left = padding_w / 2;
        if (dst_box.left % 2 != 0) {
            dst_box.left -= dst_box.left % 2;
            if (dst_box.left < 0) {
                dst_box.left = 0;
            }
        }
        dst_box.right = dst_box.left + resize_w - 1;
        _left_offset = dst_box.left;
    }
    printf("scale=%f dst_box=(%d %d %d %d) allow_slight_change=%d _left_offset=%d _top_offset=%d padding_w=%d padding_h=%d\n",
        scale, dst_box.left, dst_box.top, dst_box.right, dst_box.bottom, allow_slight_change,
        _left_offset, _top_offset, padding_w, padding_h);

    //set offset and scale
    if(letterbox != NULL){
        letterbox->scale = scale;
        letterbox->x_pad = _left_offset;
        letterbox->y_pad = _top_offset;
    }
    // alloc memory buffer for dst image,
    // remember to free
    if (dst_image->virt_addr == NULL && dst_image->fd <= 0) {
        int dst_size = get_image_size(dst_image);
        dst_image->virt_addr = (uint8_t *)malloc(dst_size);
        if (dst_image->virt_addr == NULL) {
            printf("malloc size %d error\n", dst_size);
            return -1;
        }
    }
    ret = convert_image(src_image, dst_image, &src_box, &dst_box, color);
    return ret;
}