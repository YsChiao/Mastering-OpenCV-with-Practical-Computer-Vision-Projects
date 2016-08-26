#ifndef CARTOON_H
#define CARTOON_H


#include <opencv2\core\core.hpp>

int edgedImage(cv::Mat& src, cv::Mat& dst);
int bilateralImage(cv::Mat& src, cv::Mat& dst, int factor);
int cartoonifyImage(cv::Mat src, cv::Mat& dst);


int scharrImageOpenCL(cv::Mat& src, cv::Mat& dst);
int cartoonifyImageOpenCL(cv::Mat& src, cv::Mat& dst);
int colorFaceOpenCL(cv::Mat& src, cv::Mat& dst);

int skinColorChangerOpenCL(cv::Mat& src, cv::Mat& dst);


int imagePlot(cv::Mat& result, std::initializer_list <cv::Mat> list, int row, int column);



#endif
