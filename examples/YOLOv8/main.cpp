#include "YOLOv8.hpp"

const char* MODEL_PATH = "model/yolo8s-seg.rknn";

int main(int argc, char* argv[])
{
    cv::Mat image;
    YOLOv8 yolo;
    std::vector<OBJECT> objects;
    switch (argc)
    {
    case 3:
        image = cv::imread(argv[2]);
        yolo.Initialize(argv[1]);
        objects = yolo.Detect(image);
        for (int i = 0; i < objects.size(); i++)
        {
            objects[i].Draw(image);
        }
        cv::imwrite("out.png", image);
        break;
    default:
        puts("Usage: %s <model_path> <image path>");
        //yolo.Initialize(MODEL_PATH);
        break;
    }
}