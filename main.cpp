#include "convolution2d.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

int main(int argc, const char* argv[])
{
	cv::Mat src = cv::imread("lena.jpg", cv::IMREAD_GRAYSCALE);
	if(src.empty())
	{
		std::cerr << "could not load image." << std::endl;
		return -1;
	}

	cv::Mat dst(src.size(), src.type(), cv::Scalar(0));

	// kernel
	const int kernel_size = 3;
	cv::Mat kernel = 
		(cv::Mat_<float>(kernel_size, kernel_size) 
			<<
			1.0f / 9, 1.0f / 9, 1.0f / 9,
			1.0f / 9, 1.0f / 9, 1.0f / 9,
			1.0f / 9, 1.0f / 9, 1.0f / 9
		);

	const int loop_num = 10;

	// convolution
	double time = launch_convolution2d(src, dst, kernel, loop_num);

	std::cout << "time = " << time << " ms." << std::endl;

	// display image
	cv::imshow("src", src);
	cv::imshow("dst", dst);
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}
