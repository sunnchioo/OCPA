#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <cuda.h>
#include <cudnn.h>
#include "cuda_fp16.h"

using namespace std;

#define CUDA_CALL(f)                                           \
    {                                                          \
        cudaError_t err = (f);                                 \
        if (err != cudaSuccess)                                \
        {                                                      \
            std::cout                                          \
                << "    Error occurred: " << err << std::endl; \
            std::exit(1);                                      \
        }                                                      \
    }

#define CUDNN_CALL(f)                                          \
    {                                                          \
        cudnnStatus_t err = (f);                               \
        if (err != CUDNN_STATUS_SUCCESS)                       \
        {                                                      \
            std::cout                                          \
                << "    Error occurred: " << err << std::endl; \
            std::exit(1);                                      \
        }                                                      \
    }

void print(const float *data, int n, int c, int h, int w)
{
    std::vector<float> buffer(1 << 20);
    CUDA_CALL(cudaMemcpy(
        buffer.data(), data,
        n * c * h * w * sizeof(float),
        cudaMemcpyDeviceToHost));
    int a = 0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < c; ++j)
        {
            std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
            for (int k = 0; k < h; ++k)
            {
                for (int l = 0; l < w; ++l)
                {
                    std::cout << std::setw(4) << std::right << buffer[a];
                    ++a;
                }
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
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
        ifstream kernel_data("dataset/resnetdataset_all/kernel/" + name);
        for (int j = i * (*kernel_width * *kernel_height); j < (i + 1) * (*kernel_width * *kernel_height); j++)
            kernel_data >> kernel[j];
        kernel_data.close();
    }

    return kernel;
}

half *LoadvggKernel(string name, int *&kernel_width, int *&kernel_height, int batch_size)
{
    float temp;
    *kernel_width = *kernel_height = 3;
    half *kernel = new half[*kernel_width * *kernel_height * batch_size];
    for (int i = 0; i < batch_size; i++)
    {
        ifstream kernel_data("/home/lfa/fsy/syt/conv_pool/dataset/vggdata/kernel/" + name +".txt");
        for (int j = i * (*kernel_width * *kernel_height); j < (i + 1) * (*kernel_width * *kernel_height); j++)
        {    
            kernel_data >> temp;
            kernel[j]=__float2half(temp);
        }
        kernel_data.close();
    }

    return kernel;
}

float *LoadConvWeight(int *&fea_width, int *&fea_height, int batch_size, int index)
{
    string *name = new string[49];

    ifstream fea_name("dataset/resnetdataset_all/feature_name.txt");
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
        ifstream fea_data("dataset/resnetdataset_all/feature/" + name[index]);
        for (int j = i * (*fea_width * *fea_height); j < (i + 1) * (*fea_width * *fea_height); j++)
            fea_data >> feature[j];
        fea_data.close();
    }

    return feature;
}

half *LoadVggConvWeigth(int *&fea_width, int *&fea_height, int batch_size, int index)
{
    string *name = new string[16];

    ifstream fea_name("/home/lfa/fsy/syt/conv_pool/dataset/vggdata/all_conv_name.txt");
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
    float temp;
    half *feature = new half[batch_size * *fea_width * *fea_height];

    for (int i = 0; i < batch_size; i++)
    {
        ifstream fea_data("/home/lfa/fsy/syt/conv_pool/dataset/vggdata/feature/" + name[index]);
        for (int j = i * (*fea_width * *fea_height); j < (i + 1) * (*fea_width * *fea_height); j++)
        {    
            fea_data >> temp;
            feature[j]=__float2half(temp);
        }
        fea_data.close();
    }

    return feature;
}

half *LoadspConvWeigth(int *&fea_width, int *&fea_height, int batch_size, int index)
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
    float temp;
    for (int i = 0; i < batch_size; i++)
    {
        ifstream fea_data("dataset/sparsity/dataset/" + name[index]);
        for (int j = i * (*fea_width * *fea_height); j < (i + 1) * (*fea_width * *fea_height); j++)
            fea_data >> temp;
            feature[j] = __float2half(temp);
        fea_data.close();
    }

    return feature;
}

int main(int argc, char *argv[])
{
    int batch_size = atoi(argv[1]);
    // load kernel file name
    // string kername_name[49];
    // ifstream open_kernel("/home/syt/conv_pool/conv_pool/dataset/resnetdataset_all/kernel_name.txt");
    string kername_name[9];
    ifstream open_kernel("dataset/sparsity/dataset/sparsity_name.txt");
    for (int i = 0; i < 9; i++)
    {
        open_kernel >> kername_name[i];
    }
    open_kernel.close();

    ofstream time_file("ECR/ECR/time_sparsity/batchsize" + argv[1] + string(".txt"));
    //  ofstream time_file(string("/home/syt/conv_pool/conv_pool/ECR/cudnn/time_gemm/batchsize") + argv[1] + string(".txt"));

    for (int t = 0; t < 9; t++)
    {
        cudnnHandle_t cudnn;
        CUDNN_CALL(cudnnCreate(&cudnn));

        // input
        // int batch_size = 2;
        int *fea_width_ = new int;
        int *fea_height_ = new int;
        half *matrix;
        // matrix = LoadConvWeight(fea_width_, fea_height_, batch_size, i);
        matrix = LoadVggConvWeigth(fea_width_, fea_height_, batch_size, t);
        // matrix = LoadspConvWeigth(fea_width_, fea_height_, batch_size, i);

        const int in_n = batch_size;
        const int in_c = 1;
        const int in_h = *fea_width_;
        const int in_w = *fea_height_;
        const int in_size = in_h * in_w * in_c * in_n;

        // cout << "in:" << in_n << " " << in_c << " " << in_h << " " << in_w << endl;

        // filter
        // int stride = 1;
        int *kernel_width_ = new int;
        int *kernel_height_ = new int;
        half *kernel

        kernel = LoadspConvWeigth(kername_name[t], kernel_width_, kernel_height_, batch_size);
        const int filt_k = 1;
        const int filt_c = 1;
        const int filt_h = *kernel_width_;
        const int filt_w = *kernel_height_;
        const int file_size = filt_h * filt_w * filt_c * filt_k;

        // cout << "ker:" << filt_k << " " << filt_c << " " << filt_h << " " << filt_w << endl;

        // 记录时间
        cudaEvent_t start, stop;
        float elapsed_time = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        cudnnTensorDescriptor_t in_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
            in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
            in_n, in_c, in_h, in_w));

            half *in_data;
        CUDA_CALL(cudaMalloc(
            &in_data, in_n * in_c * in_h * in_w * sizeof(float)/2));

        cudnnFilterDescriptor_t filt_desc;
        CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
        CUDNN_CALL(cudnnSetFilter4dDescriptor(
            filt_desc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
            filt_k, filt_c, filt_h, filt_w));

        half *filt_data;
        CUDA_CALL(cudaMalloc(
            &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)/2));

        // convolution
        const int pad_h = 0;
        const int pad_w = 0;
        const int str_h = 1;
        const int str_w = 1;
        const int dil_h = 1;
        const int dil_w = 1;

        cudnnConvolutionDescriptor_t conv_desc;
        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
        CUDNN_CALL(cudnnSetConvolution2dDescriptor(
            conv_desc,
            pad_h, pad_w, str_h, str_w, dil_h, dil_w,
            CUDNN_CONVOLUTION, CUDNN_DATA_HALF));
        // cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);
        // output
        int out_n;
        int out_c;
        int out_h;
        int out_w;

        CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
            conv_desc, in_desc, filt_desc,
            &out_n, &out_c, &out_h, &out_w));
        // cout << "out:" << out_n << " " << out_c << " " << out_h << " " << out_w << endl;

        cudnnTensorDescriptor_t out_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
            out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
            out_n, out_c, out_h, out_w));

        half *out_data;
        CUDA_CALL(cudaMalloc(
            &out_data, out_n * out_c * out_h * out_w * sizeof(float)/2));

        // algorithm
        cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

        // = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
        // = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        // = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        // = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
        // = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;

        // CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
        //     cudnn,
        //     in_desc, filt_desc, conv_desc, out_desc,
        //     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

        // workspace
        size_t ws_size;
        CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

        half *ws_data;
        CUDA_CALL(cudaMalloc(&ws_data, ws_size));

        // perform
        float alpha = 1.0;
        float beta = 0.0;

        /*
        float input_data[25 * 3] = {0, 0, 15, 8, 0, 22, 0, 0, 23, 0, 0, 8, 0, 0, 0, 0, 0, 19, 0, 0, 10, 4, 0, 0, 22,
                                    0, 0, 15, 8, 0, 22, 0, 0, 23, 0, 0, 8, 0, 0, 0, 0, 0, 19, 0, 0, 10, 4, 0, 0, 22,
                                    0, 0, 15, 8, 0, 22, 0, 0, 23, 0, 0, 8, 0, 0, 0, 0, 0, 19, 0, 0, 10, 4, 0, 0, 22};
        float kernel_data[9 * 3] = {0, 1, 0, 1, 0, 1, 0, 1, 0,
                                    0, 1, 0, 1, 0, 1, 0, 1, 0,
                                    0, 1, 0, 1, 0, 1, 0, 1, 0};
        */

        cudaMemcpy(in_data, matrix, in_size * sizeof(float)/2, cudaMemcpyHostToDevice);
        cudaMemcpy(filt_data, kernel, file_size * sizeof(float)/2, cudaMemcpyHostToDevice);

        CUDNN_CALL(cudnnConvolutionForward(
            cudnn,
            &alpha, in_desc, in_data, filt_desc, filt_data,
            conv_desc, algo, ws_data, ws_size,
            &beta, out_desc, out_data));

        // results
        // std::cout << "in_data:" << std::endl;
        // print(in_data, in_n, in_c, in_h, in_w);

        // std::cout << "filt_data:" << std::endl;
        // print(filt_data, filt_k, filt_c, filt_h, filt_w);

        // std::cout << "out_data:" << std::endl;
        // print(out_data, out_n, out_c, out_h, out_w);

        int result_size = out_n * out_c * out_h * out_w;
        half *result = new half[result_size];
        cudaMemcpy(result, out_data, result_size * sizeof(float)/2, cudaMemcpyDeviceToHost);

        // finalizing
        CUDA_CALL(cudaFree(ws_data));
        CUDA_CALL(cudaFree(out_data));
        CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
        CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
        CUDA_CALL(cudaFree(filt_data));
        CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
        CUDA_CALL(cudaFree(in_data));
        CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
        CUDNN_CALL(cudnnDestroy(cudnn));

        // 计算时间
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        // cout << elapsed_time << endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // 写入时间
        // ofstream time_file("time/time_batchsize1.txt", ios::app);
        time_file << elapsed_time << endl;
        // time_file.close();
        // 结果写入文件
        // string file_name = "/home/lfa/fsy/syt/conv_pool/ECR/cudnn/output_vgg/output" + to_string(i);
        // // string file_name = "/home/syt/conv_pool/conv_pool/ECR/cudnn/out_gemm/output" + to_string(i);

        // ofstream output_file(file_name);
        // for (int i = 0; i < result_size; i++)
        // {
        //     output_file << result[i] << " ";
        //     if ((i + 1) % out_w == 0)
        //         output_file << "\n";
        // }
        // output_file.close();
        // cout << "res:" << result[1] << endl;
        memset(result,0,sizeof(result));
        free(result);
    }

    time_file.close();

    return 0;
}