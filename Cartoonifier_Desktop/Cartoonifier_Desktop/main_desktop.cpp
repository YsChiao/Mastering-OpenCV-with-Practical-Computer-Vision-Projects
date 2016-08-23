#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>
#include <time.h>
#include "cartoon.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	int cameraNumber = 1;
	if (argc > 1)
	{
		cameraNumber = atoi(argv[1]);
	}

	// Get access to the camera
	cv::VideoCapture camera;
	camera.open(cameraNumber);
	if (!camera.isOpened()) 
	{
		std::cerr << "EROOR£ºCould not access the camera or video!" << std::endl;
		exit(1);
	}

	// Try to set the camera resolution
	camera.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	// Number of frames to capture
	int num_frames = 12;
	int count = 0;
	double fps = 0;
	time_t start, end;


	while (true)
	{ 

		// Grabe the next camera frame.
		cv::Mat cameraFrame;
		camera >> cameraFrame;
		if (cameraFrame.empty())
		{
			std::cerr << "ERROR: Couldn't grab a camera frame." << std::endl;
			exit(1);
		}

		// Create a blank output image,  that we will draw onto
		cv::Mat displayedFrame(cameraFrame.size(), CV_8UC3);
		//cartoonifyImage(cameraFrame, displayedFrame); // CPU implementation
		//cartoonifyImageOpenCL(cameraFrame, displayedFrame); // OPENCL 
		colorFaceOpenCL(cameraFrame, displayedFrame); 

		// Calculate fps for video processing
		// fps counter begin
		if (count == 0)
		{
			time(&start);
		} // fps counter end

		time(&end);
		count++;
		double sec = difftime(end, start);
		fps = count / sec;
		if (count > 10)
		{
			std::string fps_str = "fps: " + std::to_string(int(fps)); 
			cv::putText(displayedFrame, fps_str, cv::Point(550, 450), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 200, 200), 1, CV_AA);

			std::cout << fps << " fps" << std::endl;
		}
		if (count == (INT_MAX - 1000))
		{
			count = 0;
		}

		// Display the processed image onto the screen
		cv::imshow("Cartoonifier", displayedFrame);
		//std::cout << "Frames displayed : " << count << std::endl;

		// IMPORTANT: Wait for at least 20 milliseconds,
		// so that the image can be displayed on the screen!
		// Also checks if a key was pressed in the GUI window.
		// Not that it should be a "char" to support Linux.
		char keypress = cv::waitKey(20); // Need this to see anything!
		if (keypress == 27 || keypress == 113) // Escape key
		{ // Quit the program!
			camera.release();
			cv::destroyAllWindows();
			break;
		} // end while.
	}
	
	return 0;
}