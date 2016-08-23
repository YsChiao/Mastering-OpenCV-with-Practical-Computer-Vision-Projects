#ifndef CARTOON_H
#define CARTOON_H


#include <opencv2\core\core.hpp>

int edgedImage(cv::Mat& src, cv::Mat& dst);
int bilateralImage(cv::Mat& src, cv::Mat& dst, int factor);
int cartoonifyImage(cv::Mat src, cv::Mat& dst);


int scharrImageOpenCL(cv::Mat& src, cv::Mat& dst);
int cartoonifyImageOpenCL(cv::Mat& src, cv::Mat& dst);
int colorFaceOpenCL(cv::Mat& src, cv::Mat& dst);

#endif
