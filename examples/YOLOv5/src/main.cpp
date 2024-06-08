#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>

#include "YOLOv5.hpp"

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Usage: %s <rknn model> <input_image_path> <resize/letterbox> <output_image_path>\n", argv[0]);
        return -1;
    }
    YOLOv5 yolo;
    if (yolo.Initialize(argv[1]))
    {
        cv::Mat image;
        image = cv::imread(argv[2]);
        yolo.Infer(image);
    }
    return 0;
}