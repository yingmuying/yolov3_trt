#include "trt_utils.h"
#include "yolo.h"
#include "image.h"
#include "GetFiles.hpp"

#include <experimental/filesystem>
#include <fstream>
#include <string>
#include <ctime>
using namespace std;
using namespace cv;

bool decode = true;

std::string configFilePath = "../data/vapd.cfg";
std::string wtsFilePath = "../data/vapd.weights";
std::string precision = "kFLOAT";
std::string enginePath = "../data/vapd-" + precision + "-kGPU-batch1" + ".engine";
uint class_num = 1;
float probThresh = 0.2;


int main(int argc, char** argv)
{

    std::unique_ptr<Yolo> inferNet(new Yolo(configFilePath, wtsFilePath, precision, enginePath, class_num, probThresh));

    std::vector<std::string> imageList;
    getFilesName("/home/zxj/yolov3_trt/imgs", imageList);

    std::cout << "Total number of images used for inference : " << imageList.size() << std::endl;

    double t = cv::getTickCount();

    // Batched inference loop
    for (uint loopIdx = 0; loopIdx < imageList.size(); ++loopIdx)
    {
        Mat img = imread(imageList[loopIdx]);

        image im = cv_img_to_image(img);
        image sized = letterbox_image(im, inferNet->getInputW(), inferNet->getInputH());   //调整尺寸
        uchar *data = reinterpret_cast<uchar *>(sized.data);

        double t1 = cv::getTickCount();
        inferNet->doInference(data);
        cout<<(cv::getTickCount() - t1) * 1000.0 / cv::getTickFrequency()<< " ms\n";

        if (decode) {
            std::vector<BBoxInfo> binfo = inferNet->decodeDetections(0, im.h, im.w);
            std::vector<BBoxInfo> remaining = nmsAllClasses(inferNet->getNMSThresh(), binfo, inferNet->getClassNum());
            for (auto b : remaining) {
                printPredictions(b);

                int H = img.rows;
                int W = img.cols;
                float x1 = b.box.x1;
                float y1 = b.box.y1;
                float x2 = b.box.x2;
                float y2 = b.box.y2;

                if(x1 < 0) x1 = 0.0;
                if(y1 < 0) y1 = 0.0;
                if(x2 >= W) x2 = W - 1.0;
                if(y2 >= H) y2 = H - 1.0;

                cv::rectangle(img, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255), 2);

                float x = (x1 + x2) / (2.0 * W);
                float y = (y1 + y2) / (2.0 * H);
                float w = (x2 - x1) * 1.0 / W;
                float h = (y2 - y1) * 1.0 / H;
                cout<<x<<" "<<y<<" "<<w<<" "<<h<<endl;
            }
            imshow("", img);
            waitKey();
        }
    }
    cout<<"It takes "<<(cv::getTickCount() - t) * 1000.0 / cv::getTickFrequency() / imageList.size() << " ms per inference\n";

    std::cout << std::endl
              << "Network Type : YoloV3\n" << "Precision : " << precision
              << std::endl;

    return 0;
}
