#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

double launch_convolution2dGpu
(
	const cv::cuda::GpuMat& src, 
	cv::cuda::GpuMat& dst, 
	const cv::cuda::GpuMat& kernel, 
	const int loop_num
);

