#include "cartoon.h"
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ocl\ocl.hpp>
#include <time.h>
#include <iostream>

int cartoonifyImageOpenCL(cv::Mat src, cv::Mat& dst)
{
	// Convert image containter 
	cv::ocl::oclMat srcImage;
	srcImage.upload(src);

	cv::ocl::oclMat gray;
	cv::ocl::cvtColor(srcImage, gray, CV_BGR2GRAY);

	// SET MEDIAN FILTER SIZE
	const int MEDIAN_BLUR_FILTER_SIZE = 5;
	cv::ocl::medianFilter(gray, gray, MEDIAN_BLUR_FILTER_SIZE);

	cv::ocl::oclMat edges;
	const int LPALACIAN_FILTER_FIZE = 3;
	cv::ocl::Laplacian(gray, edges, CV_8U, LPALACIAN_FILTER_FIZE);

	// Apply binary threshold
	cv::ocl::oclMat mask;
	const int EDGES_THREASHOLD = 15;
	cv::ocl::threshold(edges, mask, EDGES_THREASHOLD, 255, CV_THRESH_BINARY_INV);

	// Reduce the size of original image by a factor of four
	int factor = 4;
	cv::Size size = srcImage.size();
	cv::Size smallSize;
	smallSize.width = size.width / (factor / 2);
	smallSize.height = size.height / (factor / 2);

	cv::Mat smallImage = cv::Mat(smallSize, CV_8UC3);

	cv::ocl::oclMat smallOclImage = cv::ocl::oclMat(smallSize, CV_8UC3);
	cv::ocl::resize(srcImage, smallOclImage, smallSize, 0, 0, CV_INTER_LINEAR);

	// Bilateral Filter
	cv::ocl::oclMat tmp = cv::ocl::oclMat(smallSize, CV_8UC3);
	int repetitions = 7; // Repetitions for strong cartoon effect.
	for (int i = 0; i < repetitions; i++)
	{
		int ksize = 5; // Filter size. Has a large effect on speed.
		double sigmaColor = 10; // Filer color strength;
		double sigmaSpace = 10; // Spatial strength.  Affects speed.
		cv::ocl::bilateralFilter(smallOclImage, tmp, ksize, sigmaColor, sigmaSpace);
		cv::ocl::bilateralFilter(tmp, smallOclImage, ksize, sigmaColor, sigmaSpace);
	}

	// Expand the image back to the original size
	cv::ocl::oclMat bigOclImage;
	cv::ocl::resize(smallOclImage, bigOclImage, size, 0, 0, CV_INTER_LINEAR);
	
	cv::ocl::oclMat image = cv::ocl::oclMat(size, CV_8UC3);
	image.setTo(0);
	bigOclImage.copyTo(image, mask);
	image.download(dst);

	return 0;
}



int cartoonifyImage(cv::Mat src, cv::Mat& dst)
{
	cv::Mat gray;	
	cv::cvtColor(src, gray, CV_BGR2GRAY);

	// SET MEDIAN FILTER SIZE
	const int MEDIAN_BLUR_FILTER_SIZE = 5;
	cv::medianBlur(gray, gray, MEDIAN_BLUR_FILTER_SIZE);

	cv::Mat edges;
	const int LPALACIAN_FILTER_FIZE = 3;
	cv::Laplacian(gray, edges, CV_8U, LPALACIAN_FILTER_FIZE);

	// Apply binary threshold
	cv::Mat mask;
	const int EDGES_THREASHOLD = 15;
	cv::threshold(edges, dst, EDGES_THREASHOLD, 255, CV_THRESH_BINARY_INV);

	// Reduce the size of original image by a factor of four
	int factor = 4;
	cv::Size size = src.size();
	cv::Size smallSize;
	smallSize.width = size.width / (factor / 2);
	smallSize.height = size.height / (factor / 2);

	cv::Mat smallImage = cv::Mat(smallSize, CV_8UC3);
	cv::resize(src, smallImage, smallSize, 0, 0, CV_INTER_LINEAR);

	// Bilateral Filter
	cv::Mat tmp = cv::Mat(smallSize, CV_8UC3);
	int repetitions = 7; // Repetitions for strong cartoon effect.
	for (int i = 0; i < repetitions; i++)
	{
		int ksize = 5; // Filter size. Has a large effect on speed.
		double sigmaColor = 10; // Filter color strength.
		double sigmaSpace = 10; // Spatial strength. Affects speed.
		cv::bilateralFilter(smallImage, tmp, ksize, sigmaColor, sigmaSpace);
		cv::bilateralFilter(tmp, smallImage, ksize, sigmaColor, sigmaSpace);
	}

	// Expand the image back to the original size
	cv::Mat bigImg;
	cv::resize(smallImage, bigImg, size, 0, 0, CV_INTER_LINEAR);
	dst.setTo(0);
	bigImg.copyTo(dst, mask);

	return 0;

}


int edgedImage(cv::Mat& src, cv::Mat& dst)
{
	cv::Mat gray;
	cv::cvtColor(src, gray, CV_BGR2GRAY);

	// Set median filter size
	const int MEDIAN_BLUR_FILTER_SIZE = 11;
	cv::medianBlur(gray, gray, MEDIAN_BLUR_FILTER_SIZE);

	cv::Mat edges;
	const int LAPLACIAN_FILTER_SIZE = 5;
	cv::Laplacian(gray, edges, CV_8U, LAPLACIAN_FILTER_SIZE);

	// Apply binary threshold
	//cv::Mat mask;
	const int EDGES_THREASHOLD = 40;
	cv::threshold(edges, dst, EDGES_THREASHOLD, 255, CV_THRESH_BINARY_INV);

	return 0;

}

int bilateralImage(cv::Mat& src, cv::Mat& dst, int factor)
{
	// Reduce the size of original image by a factor of four
	cv::Size size = src.size();
	cv::Size smallSize;
	smallSize.width = size.width / (factor / 2);
	smallSize.height = size.height / (factor / 2);

	cv::Mat smallImage = cv::Mat(smallSize, CV_8UC3);
	cv::resize(src, smallImage, smallSize, 0, 0, CV_INTER_LINEAR);

	// Bilateral Filter
	cv::Mat tmp = cv::Mat(smallSize, CV_8UC3);
	int repetitions = 7; // Repetitions for strong cartoon effect.
	for (int i = 0; i < repetitions; i++)
	{
		int ksize = 9; // Filter size. Has a large effect on speed.
		double sigmaColor = 9; // Filter color strength.
		double sigmaSpace = 7; // Spatial strength. Affects speed.
		cv::bilateralFilter(smallImage, tmp, ksize, sigmaColor, sigmaSpace);
		cv::bilateralFilter(tmp, smallImage, ksize, sigmaColor, sigmaSpace);
	}

	// Expand the image back to the original size
	cv::Mat bigImg;
	cv::resize(smallImage, bigImg, size, 0, 0, CV_INTER_LINEAR);
	bigImg.copyTo(dst);


	return 0;
}




