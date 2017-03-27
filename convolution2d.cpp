#include "convolution2d.hpp"

void convolution2d
(
	const cv::Mat& src, 
	cv::Mat& dst, 
	const cv::Mat& kernel
)
{
	const int half_size = (kernel.cols / 2);
	for (int y = half_size; y < (src.rows - half_size); y++)
	{
		for (int x = half_size; x < (src.cols - half_size); x++)
		{
			float sum = 0.0f;
			for (int dy = -half_size; dy <= half_size; dy++)
			{
				for (int dx = -half_size; dx <= half_size; dx++)
				{
					sum += (kernel.ptr<float>(dy+half_size)[dx+half_size] * src.ptr<uchar>(y+dy)[x+dx]);
				}
			}
			dst.ptr<uchar>(y)[x] = sum;
		}
	}
}

double launch_convolution2d
(
	const cv::Mat& src, 
	cv::Mat& dst, 
	const cv::Mat& kernel, 
	const int loop_num
)
{
	double f = 1000.0 / cv::getTickFrequency();
	int64 start = 0, end = 0;
	double time = 0.0;
	for (int i = 0; i <= loop_num; i++)
	{
		start = cv::getTickCount();

		convolution2d(src, dst, kernel);

		end = cv::getTickCount();
		time += (i > 0) ? ((end - start) * f) : 0;
	}
	time /= loop_num;

	return time;
}
