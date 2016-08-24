#include "cartoon.h"
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ocl\ocl.hpp>
#include <time.h>
#include <iostream>

int cartoonifyImageOpenCL(cv::Mat& src, cv::Mat& dst)
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
	const int EDGES_THREASHOLD = 80;
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
	cv::threshold(edges, mask, EDGES_THREASHOLD, 255, CV_THRESH_BINARY_INV);

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


int scharrImageOpenCL(cv::Mat& src, cv::Mat& dst)
{
	cv::ocl::oclMat oclSrc = cv::ocl::oclMat(src.size(), CV_8UC3);
	oclSrc.upload(src);

	cv::ocl::oclMat oclGray;

	cv::ocl::cvtColor(oclSrc, oclGray, CV_BGR2GRAY);
	const int MEDIAN_BLUR_FILTER_SIZE = 5;
	cv::ocl::medianFilter(oclGray, oclGray, MEDIAN_BLUR_FILTER_SIZE);

	cv::ocl::oclMat edges, edges2;
	cv::ocl::Scharr(oclGray, edges, CV_8UC1, 1, 0);
	cv::ocl::Scharr(oclGray, edges2, CV_8UC1, 1, 0, -1);
	edges += edges2;
	const int EVIL_EDGE_THREASHOLD = 12;
	cv::ocl::oclMat mask;
	cv::ocl::threshold(edges, mask, EVIL_EDGE_THREASHOLD, 255, CV_THRESH_BINARY_INV);
	cv::ocl::medianFilter(mask, mask, 3);

	// Reduce the size of original image by a factor of four
	int factor = 4;
	cv::Size size = oclSrc.size();
	cv::Size smallSize;
	smallSize.width = size.width / (factor / 2);
	smallSize.height = size.height / (factor / 2);

	cv::Mat smallImage = cv::Mat(smallSize, CV_8UC3);

	cv::ocl::oclMat smallOclImage = cv::ocl::oclMat(smallSize, CV_8UC3);
	cv::ocl::resize(oclSrc, smallOclImage, smallSize, 0, 0, CV_INTER_LINEAR);

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


int colorFaceOpenCL(cv::Mat& src, cv::Mat& dst) 
{
	// Draw the color face onto a black background.
	cv::Mat faceOutline = cv::Mat(src.size(), CV_8UC3, double(0));
	//faceOutline.setTo(0);
	cv::Scalar color = CV_RGB(255, 0, 0);  // Yellow.
	int thickness = 1;
	// Use 70% of the screen height as the face height.
	int sw = src.size().width;
	int sh = src.size().height;
	int faceH = sh / 2 * 70 / 100; // "faceH" is the radius of the elipse.
	// Scale the width to be the same shape for any screen width
	int faceW = faceH * 72 / 100;
	// Draw the face outline.
	cv::ellipse(faceOutline, cv::Point(sw / 2, sh / 2), cv::Size(faceW, faceH), 0, 0, 360, color, thickness, CV_AA);


	// Draw the eye outlines, as 2 arcs per eye.
	int eyeW = faceW * 23 / 100;
	int eyeH = faceH * 11 / 100;
	int eyeX = faceW * 48 / 100;
	int eyeY = faceH * 13 / 100;
	cv::Size eyeSize = cv::Size(eyeW, eyeH);
	// Set the angle  and shift for the eye half ellipses.
	int eyeA = 15; // angle in degrees.
	int eyeYshift = 11;

	// Draw the top of the right eye
	cv::ellipse(faceOutline, cv::Point(sw / 2 - eyeX, sh / 2 - eyeY), eyeSize, 0, 180 + eyeA, 360 - eyeA, color, thickness, CV_AA);
	// Draw the bottom of the right eye.
	cv::ellipse(faceOutline, cv::Point(sw / 2 - eyeX, sh / 2 - eyeY - eyeYshift), eyeSize, 0, 0 + eyeA, 180 - eyeA, color, thickness, CV_AA);
	// Draw the top of the left eye.
	cv::ellipse(faceOutline, cv::Point(sw / 2 + eyeX, sh / 2 - eyeY), eyeSize, 0, 180 + eyeA, 360 - eyeA, color, thickness, CV_AA);
	// Draw the bottom of the left eye.
	cv::ellipse(faceOutline, cv::Point(sw / 2 + eyeX, sh / 2 - eyeY - eyeYshift), eyeSize, 0, 0 + eyeA, 180 - eyeA, color, thickness, CV_AA);


	// Draw the bottom lip of the mouth.
	int mouthY = faceH * 48 / 100;
	int mouthW = faceW * 45 / 100;
	int mouthH = faceH * 6 / 100;
	cv::ellipse(faceOutline, cv::Point(sw / 2, sh / 2 + mouthY), cv::Size(mouthW, mouthH), 0, 0, 180, color, thickness, CV_AA);

	// Draw anti-aliased text
	int fontFace = CV_FONT_HERSHEY_COMPLEX;
	float fontScale = 0.5f;
	int fontThickness = 1;
	std::string szMsg = "Put your face here";
	cv::putText(faceOutline, szMsg, cv::Point(sw * 23 / 100, sh * 10 / 100), fontFace, fontScale, color, fontThickness, CV_AA);


	// Use OPENCL, but with even slow speed. upload and download need more time than OPENCL computing
	//cv::ocl::oclMat srcOcl = cv::ocl::oclMat(src.size(), CV_8UC3);
	//cv::ocl::oclMat faceOutlineOcl = cv::ocl::oclMat(faceOutline.size(), CV_8UC3);
	//cv::ocl::oclMat dstOcl = cv::ocl::oclMat(dst.size(), CV_8UC3);
	//srcOcl.upload(src);
	//faceOutlineOcl.upload(faceOutline);
	//cv::ocl::addWeighted(srcOcl, 1.0, faceOutlineOcl, 0.7, 0, dstOcl);
	//dstOcl.download(dst);


	cv::addWeighted(src, 1.0, faceOutline, 0.8, 0, dst, CV_8UC3);
	return 0;

}


int skinColorChangerOpenCL(cv::Mat& src, cv::Mat& dst)
{

	//// OPENCL 
	// Convert image containter 
	cv::ocl::oclMat srcOclImage;
	srcOclImage.upload(src);

	//// Convert to gray image;
	//cv::ocl::oclMat grayOclImage;
	//cv::ocl::cvtColor(srcOclImage, grayOclImage, CV_BGR2GRAY);

	//// SET MEDIAN FILTER SIZE
	//const int MEDIAN_BLUR_FILTER_SIZE = 5;
	//cv::ocl::medianFilter(grayOclImage, grayOclImage, MEDIAN_BLUR_FILTER_SIZE);

	//// Calculate edges
	//cv::ocl::oclMat edgesOclImage;
	//const int LPALACIAN_FILTER_SIZE = 3;
	//cv::ocl::Laplacian(grayOclImage, edgesOclImage, CV_8U, LPALACIAN_FILTER_SIZE);


	// ======================= Non-OPENCL image smoothing and edges detection =============================================
	// Convert to gray
	cv::Size size = src.size();
	cv::Mat grayImage = cv::Mat(size, CV_8UC1);
	cv::cvtColor(src, grayImage, CV_BGR2GRAY);

	// Median Filter and Lapalacian Filter
	cv::medianBlur(grayImage, grayImage, 7);
	cv::Mat edges;
	cv::Laplacian(grayImage, edges, CV_8U, 5);

	// Mask of edges
	int sh = size.height;
	int sw = size.width;
	cv::Mat mask, maskPlusBorder;
	maskPlusBorder = cv::Mat::zeros(sh + 2, sw + 2, CV_8UC1); // black-white mask.
	mask = maskPlusBorder(cv::Rect(1, 1, sw, sh)); // maks is in maskPlusBorder.
	
	// Apply binary threshold
	const int EDGES_THREASHOLD = 100;
	cv::threshold(edges, mask, EDGES_THREASHOLD, 255, CV_THRESH_BINARY_INV);

	// Reduce the size of original image by a factor of four
	int factor = 4;
	cv::Size smallSize;
	smallSize.width = size.width / (factor / 2);
	smallSize.height = size.height / (factor / 2);

	cv::ocl::oclMat smallOclImage = cv::ocl::oclMat(smallSize, CV_8UC3);
	cv::ocl::resize(srcOclImage, smallOclImage, smallSize, 0, 0, CV_INTER_LINEAR);

	// Bilateral Filter, using OPENCL to speed up
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

	// OPENCL download
	cv::Mat smallImage = cv::Mat(smallSize, CV_8UC3);
	smallOclImage.download(smallImage);
	cv::Mat bigImage;
	cv::resize(smallImage, bigImage, size, 0, 0, CV_INTER_LINEAR);
	dst.setTo(0);
	bigImage.copyTo(dst, mask);

	// ================================= Showing the user where to put their face =====================================================================
	// Draw a color face onto a black background.
	cv::Mat faceOutline = cv::Mat::zeros(size, CV_8UC3);
	cv::Scalar color = CV_RGB(255, 0, 0);  // Yellow.
	int thickness = 4;
	// Use 70% of the screen height as the face height.
	int faceH = sh / 2 * 70 / 100; // "faceH" is the radius of the elipse.
	// Scale the width to be the same shape for any screen width
	int faceW = faceH * 72 / 100;
	// Draw the face outline.
	cv::ellipse(faceOutline, cv::Point(sw / 2, sh / 2), cv::Size(faceW, faceH), 0, 0, 360, color, thickness, CV_AA);

	// Draw the eye outlines, as 2 arcs per eye.
	int eyeW = faceW * 23 / 100;
	int eyeH = faceH * 11 / 100;
	int eyeX = faceW * 48 / 100;
	int eyeY = faceH * 13 / 100;
	cv::Size eyeSize = cv::Size(eyeW, eyeH);
	// Set the angle  and shift for the eye half ellipses.
	int eyeA = 15; // angle in degrees.
	int eyeYshift = 11;

	// Draw the top of the right eye
	cv::ellipse(faceOutline, cv::Point(sw / 2 - eyeX, sh / 2 - eyeY), eyeSize, 0, 180 + eyeA, 360 - eyeA, color, thickness, CV_AA);
	// Draw the bottom of the right eye.
	cv::ellipse(faceOutline, cv::Point(sw / 2 - eyeX, sh / 2 - eyeY - eyeYshift), eyeSize, 0, 0 + eyeA, 180 - eyeA, color, thickness, CV_AA);
	// Draw the top of the left eye.
	cv::ellipse(faceOutline, cv::Point(sw / 2 + eyeX, sh / 2 - eyeY), eyeSize, 0, 180 + eyeA, 360 - eyeA, color, thickness, CV_AA);
	// Draw the bottom of the left eye.
	cv::ellipse(faceOutline, cv::Point(sw / 2 + eyeX, sh / 2 - eyeY - eyeYshift), eyeSize, 0, 0 + eyeA, 180 - eyeA, color, thickness, CV_AA);

	// Draw the bottom lip of the mouth.
	int mouthY = faceH * 48 / 100;
	int mouthW = faceW * 45 / 100;
	int mouthH = faceH * 6 / 100;
	cv::ellipse(faceOutline, cv::Point(sw / 2, sh / 2 + mouthY), cv::Size(mouthW, mouthH), 0, 0, 180, color, thickness, CV_AA);

	// Draw anti-aliased text
	int fontFace = CV_FONT_HERSHEY_COMPLEX;
	float fontScale = 1.0f;
	int fontThickness = 2;
	std::string szMsg = "Put your face here";
	cv::putText(faceOutline, szMsg, cv::Point(sw * 23 / 100, sh * 10 / 100), fontFace, fontScale, color, fontThickness, CV_AA);
	cv::addWeighted(dst, 1.0, faceOutline, 0.8, 0, dst, CV_8UC3);


	// ============================================Skin detection and changer ============================================ 
	// YCrCb Image
	cv::Mat yuvImage = cv::Mat(smallSize, CV_8UC3);
	cv::cvtColor(smallImage, yuvImage, CV_BGR2YCrCb);

	// Mask of skin color
	int smallSW = smallSize.width;
	int smallSH = smallSize.height;
	cv::Mat smallMask, smallMaskPlusBorder; 
	smallMaskPlusBorder = cv::Mat::zeros(smallSH + 2, smallSW + 2, CV_8UC1); // black-white mask.
	smallMask = smallMaskPlusBorder(cv::Rect(1, 1, smallSW, smallSH)); // maks is in maskPlusBorder.
	cv::resize(edges, smallMask, smallSize); // Put edges in both of them.

	// Apply binary threshold
	cv::threshold(smallMask, smallMask, EDGES_THREASHOLD, 255, CV_THRESH_BINARY);
	cv::dilate(smallMask, smallMask, cv::Mat());
	cv::erode(smallMask, smallMask, cv::Mat());

	// 6 Points 
	int const NUM_SKIN_POINTS = 6;
	cv::Point skinPts[NUM_SKIN_POINTS];
	skinPts[0] = cv::Point(smallSW / 2, smallSH / 2 - smallSH / 6);
	skinPts[1] = cv::Point(smallSW / 2 - smallSW / 11, smallSH / 2 - smallSH / 6);
	skinPts[2] = cv::Point(smallSW / 2 + smallSW / 11, smallSH / 2 - smallSH / 6);
	skinPts[3] = cv::Point(smallSW / 2, smallSH / 2 + smallSH / 16);
	skinPts[4] = cv::Point(smallSW / 2 - smallSW / 9, smallSH / 2 + smallSH / 16);
	skinPts[5] = cv::Point(smallSW / 2 + smallSW / 9, smallSH / 2 + smallSH / 16);
	
	//// Draw 6 points
	//for (int i = 0; i < NUM_SKIN_POINTS; i++)
	//{
	//	cv::circle(smallImage, skinPts[i], 2, cv::Scalar(255, 0, 0));
	//}

	// YCrCb
	const int LOWER_Y = 60;
	const int UPPER_Y = 80;
	const int LOWER_Cr = 25;
	const int UPPER_Cr = 15;
	const int LOWER_Cb = 20;
	const int UPPER_Cb = 15;
	cv::Scalar lowerDiff = cv::Scalar(LOWER_Y, LOWER_Cr, LOWER_Cb);
	cv::Scalar upperDiff = cv::Scalar(UPPER_Y, UPPER_Cr, UPPER_Cb);

	// Color changer
	const int CONNECTED_COMPONENTS = 4; // To fill diagonally, use 8.
	const int flags = CONNECTED_COMPONENTS | cv::FLOODFILL_FIXED_RANGE | cv::FLOODFILL_MASK_ONLY;
	cv::Mat smallEdgeMask = smallMask.clone(); // Keep a copy of the edge mask.
	
	// "maskPlusBorder" is initialized with edges to block floodFill().
	for (int i = 0; i< NUM_SKIN_POINTS - 1; i++) 
	{
		// Draw points
		cv::circle(smallImage, skinPts[i], 1, cv::Scalar(255, 0, 0));
		// Color detect
		cv::floodFill(yuvImage, smallMaskPlusBorder, skinPts[i], cv::Scalar(), NULL, lowerDiff, upperDiff, flags);
	}

	smallMask -= smallEdgeMask;
	//smallImage.copyTo(dst, mask);
	cv::add(smallImage, cv::Scalar(0, 70, 0), smallImage, smallMask);

	// Expand the image back to the original size
	cv::resize(smallImage, dst, size, 0, 0, CV_INTER_LINEAR);
	//dst = smallMask;
	return 0;

}



