#include "convolution2d.hpp"
#include "convolution2d.cuh"
#include "utility.hpp"
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
	cv::Mat dstGpu(src.size(), src.type(), cv::Scalar(0));

	// kernel
	const int kernel_size = 3;
	cv::Mat kernel = 
		(cv::Mat_<float>(kernel_size, kernel_size) 
			<<
			1.0f / 9, 1.0f / 9, 1.0f / 9,
			1.0f / 9, 1.0f / 9, 1.0f / 9,
			1.0f / 9, 1.0f / 9, 1.0f / 9
		);

	cv::cuda::GpuMat d_src(src);
	cv::cuda::GpuMat d_dstGpu(src.size(), src.type(), cv::Scalar(0));
	cv::cuda::GpuMat d_kernel(kernel);

	const int loop_num = 10;

	// convolution
	double time_cpu = launch_convolution2d(src, dst, kernel, loop_num);
	double time_gpu = launch_convolution2dGpu(d_src, d_dstGpu, d_kernel, loop_num);

	std::cout << "time(cpu) = " << time_cpu << " ms." << std::endl;
	std::cout << "time(gpu) = " << time_gpu << " ms." << std::endl;

	// verify
	verify(dst, d_dstGpu);

	// display image
	cv::imshow("src", src);
	cv::imshow("dst", dst);
	d_dstGpu.download(dstGpu);
	cv::imshow("dst(gpu)", dstGpu);

	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}
