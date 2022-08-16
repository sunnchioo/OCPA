#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <fstream>
#include <assert.h>
#include <cuda_runtime.h>
#include "cuda_fp16.h"

using namespace std;

__global__ void BatchedPECR(int batch_size, half *input, half *kernel, half *output,
                            int input_width, int input_height,
                            int kernel_width, int kernel_height, int kernel_stride,
                            int conv_output_width, int conv_output_height,
                            int pooling_width, int pooling_height,
                            int pooling_output_width, int pooling_output_height, int pooling_stride,
                            int out_width, int out_height)
{
    // 从前往后推理计算
    // 计算一个 pooling result
    // int tile_size = pooling_width + (kernel_stride - 1) * (pooling_width - 1) + (kernel_width - 1);

    const int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    // const int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    const int threadId_block = (threadIdx.y * blockDim.x) + threadIdx.x;

    __shared__ half data[1024];
    __shared__ int index[1024];
    __shared__ int count[200];
    __shared__ half temp_pool[100];

    // 越界判断
    if (blockId < out_width * out_height * batch_size)
    {
        int temp = 0;
        int block_index = threadId_block * kernel_width * kernel_height;
        for (int i = 0; i < kernel_width; i++)
        {
            for (int j = 0; j < kernel_height; j++)
            {
                // unsigned int offset = (threadIdx.x % pooling_width) + (threadIdx.x / pooling_width) * input_width;
                // offset = offset + blockId
                int offset = threadIdx.x + threadIdx.y * input_width;
                offset = offset + i * input_width + j;
                offset = offset + blockIdx.y * input_width + blockIdx.x;
                offset = offset + blockIdx.z * input_width * input_height;

                if (input[offset] != __float2half(0.0))
                {
                    int kerenl_index = i * kernel_width + j;
                    data[block_index + temp] = input[offset];
                    index[block_index + temp] = kerenl_index;
                    temp++;
                }
            }
        }
        count[threadId_block] = temp;
        // 同步一个 block 的 4 个线程
        // __syncthreads();

        half temp_value = 0;
        for (int i = block_index; i < block_index + count[threadId_block]; i++)
        {
            temp_value += data[i] * kernel[index[i]];
        }
        temp_pool[threadId_block] = temp_value;
        // 同步 blcok 中计算的值
        // __syncthreads();
        __syncwarp();

        for (int reduce_stride = blockDim.x * blockDim.y / 2; reduce_stride > 0; reduce_stride >>= 1)
        {
            if (threadId_block < reduce_stride)
            {
                temp_pool[threadId_block] = (temp_pool[threadId_block] > temp_pool[threadId_block + reduce_stride]) ? temp_pool[threadId_block] : temp_pool[threadId_block + reduce_stride];
            }
        }

        output[blockId] = temp_pool[0];
    }
}

float *LoadvggKernel(string name, int *&kernel_width, int *&kernel_height, int batch_size)
{
    *kernel_width = *kernel_height = 3;
    float *kernel = new float[*kernel_width * *kernel_height * batch_size];
    for (int i = 0; i < batch_size; i++)
    {
        ifstream kernel_data("/home/lfa/fsy/syt/conv_pool/dataset/vggdata/kernel/" + name);
        for (int j = i * (*kernel_width * *kernel_height); j < (i + 1) * (*kernel_width * *kernel_height); j++)
            kernel_data >> kernel[j];
        kernel_data.close();
    }

    return kernel;
}

float *LoadVggConvWeigth(int *&fea_width, int *&fea_height, int batch_size, int index)
{
    string *name = new string[5];

    ifstream fea_name("/home/lfa/fsy/syt/conv_pool/dataset/vggdata/pooling_name");
    for (int i = 0; i < 5; i++)
    {
        fea_name >> name[i];
    }
    fea_name.close();

    int shape[5] = {224, 112, 56, 28, 14};

    *fea_width = *fea_height = shape[index];

    float *feature = new float[batch_size * *fea_width * *fea_height];

    for (int i = 0; i < batch_size; i++)
    {
        ifstream fea_data("/home/lfa/fsy/syt/conv_pool/dataset/vggdata/feature/" + name[index]);
        for (int j = i * (*fea_width * *fea_height); j < (i + 1) * (*fea_width * *fea_height); j++)
            fea_data >> feature[j];
        fea_data.close();
    }

    return feature;
}

// argv[1]=batch_size; argv[2]=matrix_width; argv[3]=matrix_height
int main(int argc, char *argv[])
{
    int batch_size = atoi(argv[1]);
    // int batch_size = 3;

    string kername_name[16];
    ifstream open_kernel("/home/lfa/fsy/syt/conv_pool/dataset/vggdata/kernel_name.txt");
    for (int i = 0; i < 16; i++)
    {
        open_kernel >> kername_name[i];
    }
    open_kernel.close();

    ofstream time_file(string("/home/lfa/fsy/syt/conv_pool/PECR/pecr/time_vgg/batchsize") + argv[1]);
    for (int t = 0; t < 5; t++)
    {
        // kernel
        // const int kernel_width = 3;
        // const int kernel_height = 3;
        int kernel_stride = 1;
        // const float kernel[9] = {0, 1, 1, 1, 0, 1, 0, 1, 0};
        // const float kernel[9] = {1, 0, 0, 1, 1, 0, 0, 0, 1};
        int *kernel_width_ = new int;
        int *kernel_height_ = new int;
        float *kernel;
        // kernel = LoadKernel(kername_name[i], kernel_width_, kernel_height_, batch_size);
        kernel = LoadvggKernel(kername_name[0], kernel_width_, kernel_height_, batch_size);
        const int kernel_width = *kernel_width_;
        const int kernel_height = *kernel_height_;

        const int kernel_size = kernel_height * kernel_width;

        half *kernel_half = new half[kernel_size];

        for(int i=0;i<kernel_size;i++)
        {
            kernel_half[i]=__float2half(kernel[i]);
        }

        // input
        /*
        float matrix[25 * 3] = {0, 0, 15, 8, 0, 22, 0, 0, 23, 0, 0, 8, 0, 0, 0, 0, 0, 19, 0, 0, 10, 4, 0, 0, 22,
                                0, 0, 15, 8, 0, 22, 0, 0, 23, 0, 0, 8, 0, 0, 0, 0, 0, 19, 0, 0, 10, 4, 0, 0, 22,
                                0, 0, 15, 8, 0, 22, 0, 0, 23, 0, 0, 8, 0, 0, 0, 0, 0, 19, 0, 0, 10, 4, 0, 0, 22};
        */
        /*
        float matrix[16 * 3] = {0, 11, 12, 0, 3, 0, 0, 0, 0, 6, 0, 7, 9, 0, 1, 0,
                                0, 11, 12, 0, 3, 0, 0, 0, 0, 6, 0, 7, 9, 0, 1, 0,
                                0, 11, 12, 0, 3, 0, 0, 0, 0, 6, 0, 7, 9, 0, 1, 0};

        // int input_width = atoi(argv[2]);
        // int input_height = atoi(argv[3]);
        */

        // ifstream read_feature("/home/syt/conv_pool/conv_pool/dataset/feature/feature_dataset/fea_0001_0000_0000_0056_0056.txt");
        // for (int i = 0; i < 56 * 56; i++)
        // {
        //     read_feature >> matrix[i];
        // }
        // for (int i = 0; i < 56 * 56; i++)
        // {
        //     matrix[1 * 56 * 56 + i] = matrix[i];
        //     matrix[2 * 56 * 56 + i] = matrix[i];
        // }
        int *fea_width_ = new int;
        int *fea_height_ = new int;
        float *matrix;
        // matrix = LoadConvWeight(fea_width_, fea_height_, batch_size, i);
        matrix = LoadVggConvWeigth(fea_width_, fea_height_, batch_size, t);

        int input_width = *fea_width_;
        int input_height = *fea_height_;
        int array_size = input_width * input_height * batch_size;

        half *feature = new half[array_size];
        // float *feature = new float[array_size];
        // memcpy(feature, matrix, array_size * sizeof(float));

        for(int i=0;i<array_size;i++)
        {
            feature[i]=__float2half(matrix[i]);
        }

        // float *feature = new float[array_size];
        // memcpy(feature, matrix, array_size * sizeof(float));

        // convolution output shape
        int conv_output_width = (input_width - kernel_width) / kernel_stride + 1;
        int conv_output_height = (input_height - kernel_height) / kernel_stride + 1;
        // int conv_output_size = conv_output_width * conv_output_height * batch_size;

        // pooling
        int pooling_width = 2;
        int pooling_height = 2;
        int pooling_stride = 1;

        // pooling result
        int pooling_output_width = (conv_output_width - pooling_width) / pooling_stride + 1;
        int pooling_output_height = (conv_output_height - pooling_height) / pooling_stride + 1;
        int pooling_output_size = pooling_output_width * pooling_output_height * batch_size;
        // cout << pooling_output_width << " " << pooling_output_height << " " << pooling_output_size << endl;
        // 记录时间
        cudaEvent_t start, stop;
        float elapsed_time = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // gpu 变量参数信息
        half *gpu_input;
        half *gpu_kernel;
        half *gpu_output;

        int gpu_input_width = input_width;
        int gpu_input_height = input_height;
        int gpu_output_width = pooling_output_width;
        int gpu_output_height = pooling_output_height;
        int gpu_output_size = pooling_output_size;

        int gpu_conv_output_width = conv_output_width;
        int gpu_conv_output_height = conv_output_height;
        int gpu_pooling_output_width = pooling_output_width;
        int gpu_pooling_output_height = pooling_output_height;

        // error status
        cudaError_t cuda_status;

        // batch 个的空间大小
        cuda_status = cudaMalloc((void **)&gpu_input, array_size * sizeof(float)/2);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "CudaMalloc1 failed! %s\n", cudaGetErrorString(cuda_status));
        }

        cuda_status = cudaMalloc((void **)&gpu_kernel, kernel_size * sizeof(float)/2);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc2 failed %s\n", cudaGetErrorString(cuda_status));
        }

        cuda_status = cudaMalloc((void **)&gpu_output, gpu_output_size * sizeof(float)/2);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc3 failed %s\n", cudaGetErrorString(cuda_status));
        }

        // 复制内存
        cuda_status = cudaMemcpy(gpu_input, feature, array_size * sizeof(float)/2, cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy1 failed %s\n", cudaGetErrorString(cuda_status));
        }

        cuda_status = cudaMemcpy(gpu_kernel, kernel, kernel_size * sizeof(float)/2, cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy2 failed %s\n", cudaGetErrorString(cuda_status));
        }

        // gup kernel function
        // each pooling result is computed by one block on GPU
        // int pooling_block = pooling_output_height * pooling_output_width;
        // int pooling_thread = pooling_height * pooling_width;

        int pooling_block_x = pooling_output_width;
        int pooling_block_y = pooling_output_height;
        int pooling_thread_x = pooling_width;
        int pooling_thread_y = pooling_height;

        dim3 grid(pooling_block_x, pooling_block_y, batch_size);
        dim3 block(pooling_thread_x, pooling_thread_y);
        BatchedPECR<<<grid, block>>>(batch_size, gpu_input, gpu_kernel, gpu_output,
                                     gpu_input_width, gpu_input_height,
                                     kernel_width, kernel_height, kernel_stride,
                                     gpu_conv_output_width, gpu_conv_output_height,
                                     pooling_width, pooling_height,
                                     gpu_pooling_output_width, gpu_pooling_output_height, pooling_stride,
                                     gpu_output_width, gpu_output_height);
        // 传回结果
        half *result = new half[gpu_output_size];
        cuda_status = cudaMemcpy(result, gpu_output, gpu_output_size * sizeof(float)/2, cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy3 failed %s\n", cudaGetErrorString(cuda_status));
        }

        // 释放空间
        cuda_status = cudaFree(gpu_input);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaFree1 failed %s\n", cudaGetErrorString(cuda_status));
        }
        cuda_status = cudaFree(gpu_kernel);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaFree2 failed %s\n", cudaGetErrorString(cuda_status));
        }
        cuda_status = cudaFree(gpu_output);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaFree3 failed %s\n", cudaGetErrorString(cuda_status));
        }

        // 计算时间
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        // cout << elapsed_time << endl;
        time_file << elapsed_time << endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // 结果写入文件
        // ofstream output_file("/home/lfa/fsy/syt/conv_pool/PECR/pecr/output/output" + to_string(i));
        // for (int i = 0; i < gpu_output_size; i++)
        // {
        //     output_file << result[i] << " ";
        //     if ((i + 1) % (pooling_output_width) == 0)
        //         output_file << "\n";
        // }
        // output_file.close();
    }
    return 0;
}