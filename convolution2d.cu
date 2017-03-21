#include "convolution2d.cuh"

#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void convolution2dGpu
(
    const cv::cudev::PtrStepSz<uchar> src,
    cv::cudev::PtrStepSz<uchar> dst,
    const cv::cudev::PtrStepSz<float> kernel
)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int half_size = (kernel.cols / 2);

    if((y >= half_size) && y < (src.rows - half_size)){
        if((x >= half_size) && (x < (src.cols - half_size))){
            float sum = 0.0f;
            for (int dy = -half_size; dy <= half_size; dy++){
                for (int dx = -half_size; dx <= half_size; dx++){
                    sum = __fadd_rn(sum, __fmul_rn(kernel.ptr(dy+half_size)[dx+half_size], src.ptr(y+dy)[x+dx]));
                }
            }
            dst.ptr(y)[x] = sum;
        }
	}
}

void launch_convolution2dGpu
(
    const cv::cuda::GpuMat& src,
    cv::cuda::GpuMat& dst,
    const cv::cuda::GpuMat& kernel
)
{
    const dim3 block(64, 2);
    const dim3 grid(cv::cudev::divUp(dst.cols, block.x), cv::cudev::divUp(dst.rows, block.y));

    convolution2dGpu<<<grid, block>>>(src, dst, kernel);

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}

double launch_convolution2dGpu
(
    const cv::cuda::GpuMat& src,
    cv::cuda::GpuMat& dst,
    const cv::cuda::GpuMat& kernel, 
    const int loop_num
)
{
    double f = 1000.0f / cv::getTickFrequency();
    int64 start = 0, end = 0;
    double time = 0.0;
    for (int i = 0; i <= loop_num; i++){
        start = cv::getTickCount();
        launch_convolution2dGpu(src, dst, kernel);
        end = cv::getTickCount();
        time += (i > 0) ? ((end - start) * f) : 0;
    }
    time /= loop_num;

    return time;
}
