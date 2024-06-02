#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>

#include "YOLOX.hpp"

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: %s <rknn model> <input_image_path> <resize/letterbox> <output_image_path>\n", argv[0]);
        return -1;
    }
    YOLOX yolox;
    if (yolox.Initialize(argv[1]))
    {
        cv::Mat image;
        image = cv::imread(argv[2]);
        yolox.Infer(image);
    }
    return 0;
}