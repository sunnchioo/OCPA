#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <fstream>
#include <assert.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void BatchedECR(int batch_size, float *input, float *kernel, float *output,
                           int input_width, int input_height,
                           int kernel_width, int kernel_height, int stride_width,
                           int out_width, int out_height)
{
    // 一个一维的 grid 处理一个矩阵 一个 block 处理矩阵的一行 二维的 grid 处理多个矩阵
    const int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    const int threadId = blockId * blockDim.x + threadIdx.x;

    int stride = stride_width;

    // 每一个 block 都有一个 shared memory
    __shared__ float F_data[5120];
    __shared__ float K_data[5120];
    __shared__ int ptr[512];
    // float F_data[51200];
    // float K_data[51200];
    // int ptr[500];

    // 越界判断
    if (threadId < batch_size * out_height * out_width)
    {
        // 构建 ecr 存储
        int temp = 0;
        for (int i = 0; i < kernel_height; i++)
        {
            for (int j = 0; j < kernel_width; j++)
            {
                // stride = 1
                // int offset = threadId + blockId * 2 + i * input_width + j;
                // offset = offset + blockIdx.y * (kernel_height - 1) * input_width;
                // offset = offset + stride;

                int offset = threadIdx.x + i * input_width + j;
                offset = offset + blockIdx.y * (kernel_height - stride) * input_width;
                offset = offset + stride * blockId * input_width + threadIdx.x * (stride - 1);

                float value = input[offset];
                float kvalue = kernel[i * kernel_width + j];

                if ((value != 0) && (kvalue != 0))
                {
                    // 一个线程 kernel_width * kernel_height 个空 从前向后填充
                    F_data[threadIdx.x * kernel_width * kernel_height + temp] = value;
                    K_data[threadIdx.x * kernel_width * kernel_height + temp] = kvalue;
                    temp++;
                }
            }
        }
        if (temp != 0)
            ptr[threadIdx.x] = temp;
        else
            ptr[threadIdx.x] = -1;

        __syncthreads();

        if (ptr[threadIdx.x] == -1)
            output[threadId] = 0;
        else
            for (int i = 0; i < ptr[threadIdx.x]; i++)
            {
                output[threadId] += F_data[threadIdx.x * kernel_width * kernel_height + i] * K_data[threadIdx.x * kernel_width * kernel_height + i];
            }
    }

    // if (threadId < out_height * out_width)
    // {
    //     for (int i = 1; i < batch_size; i++)
    //     {
    //         output[threadId] += output[threadId + i * out_height * out_width];
    //     }
    // }
}

float *LoadKernel(string name, int *&kernel_width, int *&kernel_height, int batch_size, int index)
{
    // ifstream kernel_shape("/home/syt/conv_pool/conv_pool/dataset/kernel/kernel_shape/" + name);
    // for (int i = 0; i < 2; i++)
    // {
    //     kernel_shape >> *kernel_width >> *kernel_height;
    // }
    int shape[49] = {7,
                     1, 3, 1, 1, 3, 1, 1, 3, 1,
                     1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1,
                     1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1,
                     1, 3, 1, 1, 3, 1, 1, 3, 1};
    *kernel_width = *kernel_height = shape[index];
    float *kernel = new float[*kernel_width * *kernel_height * batch_size];
    for (int i = 0; i < batch_size; i++)
    {
        ifstream kernel_data("/home/lfa/fsy/syt/conv_pool/dataset/resnetdataset_all/kernel/" + name);
        for (int j = i * (*kernel_width * *kernel_height); j < (i + 1) * (*kernel_width * *kernel_height); j++)
            kernel_data >> kernel[j];
        kernel_data.close();
    }

    return kernel;
}

float *LoadvggKernel(string name, int *&kernel_width, int *&kernel_height, int batch_size)
{
    *kernel_width = *kernel_height = 3;
    float *kernel = new float[*kernel_width * *kernel_height * batch_size];
    for (int i = 0; i < batch_size; i++)
    {
        ifstream kernel_data("dataset/vggdata/kernel/" + name + ".txt");
        for (int j = i * (*kernel_width * *kernel_height); j < (i + 1) * (*kernel_width * *kernel_height); j++)
            kernel_data >> kernel[j];
        kernel_data.close();
    }

    return kernel;
}

float *LoadConvWeight(int *&fea_width, int *&fea_height, int batch_size, int index)
{
    string *name = new string[49];

    ifstream fea_name("/home/lfa/fsy/syt/conv_pool/dataset/resnetdataset_all/feature_name.txt");
    for (int i = 0; i < 49; i++)
    {
        fea_name >> name[i];
    }
    fea_name.close();

    // int shape[11] = {224, 112, 56, 56, 56, 28, 28, 28, 14, 14, 14};
    // int shape = 28;
    int shape[49] = {224,
                     56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
                     28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
                     14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                     7, 7, 7, 7, 7, 7, 7};

    *fea_width = *fea_height = shape[index];
    // *fea_width = *fea_height = shape;

    float *feature = new float[batch_size * *fea_width * *fea_height];

    for (int i = 0; i < batch_size; i++)
    {
        ifstream fea_data("/home/lfa/fsy/syt/conv_pool/dataset/resnetdataset_all/feature/" + name[index]);
        for (int j = i * (*fea_width * *fea_height); j < (i + 1) * (*fea_width * *fea_height); j++)
            fea_data >> feature[j];
        fea_data.close();
    }

    return feature;
}

float *LoadVggConvWeigth(int *&fea_width, int *&fea_height, int batch_size, int index)
{
    string *name = new string[16];

    ifstream fea_name("dataset/vggdata/all_conv_name.txt");
    for (int i = 0; i < 16; i++)
    {
        fea_name >> name[i];
    }
    fea_name.close();

    // int shape[11] = {224, 112, 56, 56, 56, 28, 28, 28, 14, 14, 14};
    // int shape = 28;
    int shape[16] = {224, 224, 112, 112, 56, 56, 56, 56, 28, 28, 28, 28, 14, 14, 14, 14};

    *fea_width = *fea_height = shape[index];
    // *fea_width = *fea_height = shape;

    float *feature = new float[batch_size * *fea_width * *fea_height];

    for (int i = 0; i < batch_size; i++)
    {
        ifstream fea_data("dataset/vggdata/feature/" + name[index]);
        for (int j = i * (*fea_width * *fea_height); j < (i + 1) * (*fea_width * *fea_height); j++)
            fea_data >> feature[j];
        fea_data.close();
    }

    return feature;
}

float *LoadspConvWeigth(int *&fea_width, int *&fea_height, int batch_size, int index)
{
    string *name = new string[9];

    ifstream fea_name("sparsity/dataset/sparsity_name.txt");
    for (int i = 0; i < 9; i++)
    {
        fea_name >> name[i];
    }
    fea_name.close();

    // int shape[11] = {224, 112, 56, 56, 56, 28, 28, 28, 14, 14, 14};
    int shape = 28;

    // *fea_width = *fea_height = shape[index];
    *fea_width = *fea_height = shape;

    float *feature = new float[batch_size * *fea_width * *fea_height];

    for (int i = 0; i < batch_size; i++)
    {
        ifstream fea_data("sparsity/dataset/" + name[index]);
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

    // load kernel file name
    // string kername_name[49];
    string kername_name[16];
    ifstream open_kernel("dataset/vggdata/kernel_name.txt");
    // ifstream open_kernel("/home/lfa/fsy/syt/conv_pool/dataset/resnetdataset_all/kernel_name.txt");
    for (int i = 0; i < 16; i++)
    {
        open_kernel >> kername_name[i];
    }
    open_kernel.close();

        ofstream time_file("ECR/ECR/time_vgg/batchsize" + std::to_string(atoi(argv[1])) + string(".txt"));
    // resnet is 32, vgg is 16, sp is 9, resnet is 49
    for (int i = 0; i < 16; i++)
    {
        // read kernel
        int stride = 1;
        int *kernel_width_ = new int;
        int *kernel_height_ = new int;
        float *kernel;
        // kernel = LoadKernel(kername_name[i], kernel_width_, kernel_height_, batch_size, i);
        kernel = LoadvggKernel(kername_name[0], kernel_width_, kernel_height_, batch_size);
        const int kernel_width = *kernel_width_;
        const int kernel_height = *kernel_height_;
        // const int stride = 1;
        // const float kernel[9] = {0, 1, 0, 1, 0, 1, 0, 1, 0};
        // // const float kernel[9] = {1, 0, 0, 1, 1, 0, 0, 0, 1};
        const int kernel_size = kernel_height * kernel_width;

        // input
        int *fea_width_ = new int;
        int *fea_height_ = new int;
        float *matrix;
        // matrix = LoadConvWeight(fea_width_, fea_height_, batch_size, i);
        matrix = LoadVggConvWeigth(fea_width_, fea_height_, batch_size, i);
        // matrix = LoadspConvWeigth(fea_width_, fea_height_, batch_size, i);
        /***
        float matrix[25 * 3] = {0, 0, 15, 8, 0, 22, 0, 0, 23, 0, 0, 8, 0, 0, 0, 0, 0, 19, 0, 0, 10, 4, 0, 0, 22,
                                0, 0, 15, 8, 0, 22, 0, 0, 23, 0, 0, 8, 0, 0, 0, 0, 0, 19, 0, 0, 10, 4, 0, 0, 22,
                                0, 0, 15, 8, 0, 22, 0, 0, 23, 0, 0, 8, 0, 0, 0, 0, 0, 19, 0, 0, 10, 4, 0, 0, 22};
        ***/
        /***
        float matrix[16 * 3] = {0, 11, 12, 0, 3, 0, 0, 0, 0, 6, 0, 7, 9, 0, 1, 0,
                                0, 11, 12, 0, 3, 0, 0, 0, 0, 6, 0, 7, 9, 0, 1, 0,
                                0, 11, 12, 0, 3, 0, 0, 0, 0, 6, 0, 7, 9, 0, 1, 0};
        ***/
        // int input_width = atoi(argv[2]);
        // int input_height = atoi(argv[3]);
        int input_width = *fea_width_;
        int input_height = *fea_height_;
        int array_size = input_width * input_height * batch_size;
        float *feature = new float[array_size];
        memcpy(feature, matrix, array_size * sizeof(float));

        // output shape
        int output_width = (input_width - kernel_width) / stride + 1;
        int output_height = (input_height - kernel_height) / stride + 1;
        int output_size = output_width * output_height * batch_size;
        // int result_size = output_width * output_height * batch_size;

        // 记录时间
        cudaEvent_t start, stop;
        float elapsed_time = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        float *gpu_input;
        float *gpu_kernel;
        float *gpu_output;
        float gpu_input_width;
        float gpu_input_height;
        float gpu_output_width;
        float gpu_output_height;

        gpu_input_width = input_width;
        gpu_input_height = input_height;
        gpu_output_width = output_width;
        gpu_output_height = output_height;

        // cout << gpu_input_width << " " << gpu_input_height << " " << endl;
        // cout << gpu_output_width << " " << gpu_output_height << " " << endl;

        // error status
        cudaError_t cuda_status;
        // batch 个的空间大小
        cuda_status = cudaMalloc((void **)&gpu_input, array_size * sizeof(float));
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "CudaMalloc1 failed! %s\n", cudaGetErrorString(cuda_status));
        }

        cuda_status = cudaMalloc((void **)&gpu_kernel, kernel_size * sizeof(float));
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc2 failed %s\n", cudaGetErrorString(cuda_status));
        }

        cuda_status = cudaMalloc((void **)&gpu_output, output_size * sizeof(float));
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc3 failed %s\n", cudaGetErrorString(cuda_status));
        }

        // 复制内存
        cuda_status = cudaMemcpy(gpu_input, feature, array_size * sizeof(float), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy1 failed %s\n", cudaGetErrorString(cuda_status));
        }

        cuda_status = cudaMemcpy(gpu_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy2 failed %s\n", cudaGetErrorString(cuda_status));
        }

        // gup kernel function
        // each convolution result is computed by one thread(T1, T2 or T3) on GPU
        dim3 grid(gpu_output_height, batch_size);
        dim3 block(gpu_output_width);
        BatchedECR<<<grid, block>>>(batch_size, gpu_input, gpu_kernel, gpu_output,
                                      gpu_input_width, gpu_input_height,
                                      kernel_width, kernel_height, stride,
                                      gpu_output_width, gpu_output_height);
        // 传回结果
        float *result = new float[output_size];
        cuda_status = cudaMemcpy(result, gpu_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
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
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // 写入时间
        // ofstream time_file("time/time_batchsize1.txt", ios::app);
        // ofstream time_file("time/time_batchsize1.txt");
        time_file << elapsed_time << endl;
        // time_file.close();
        // 结果写入文件
        // string file_name = "/home/syt/conv_pool/conv_pool/ECR/ECR/output_vgg/output" + to_string(i);
        // string file_name = "/home/lfa/fsy/syt/conv_pool/ECR/ECR/output_vgg/output" + to_string(i);
        // ofstream output_file(file_name);
        // for (int i = 0; i < result_size; i++)
        // {
        //     output_file << result[i] << " ";
        //     if ((i + 1) % output_width == 0)
        //         output_file << "\n";
        // }
        // output_file.close();
        free(result);
    }
    time_file.close();
    return 0;
}