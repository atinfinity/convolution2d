#pragma once
#include <opencv2/core.hpp>

double launch_convolution2d
(
	const cv::Mat& src, 
	cv::Mat& dst, 
	const cv::Mat& kernel, 
	const int loop_num
);
