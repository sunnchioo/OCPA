#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <stdio.h>
#include <malloc.h>
#include <fstream>
#include <cuda.h>
#include <cudnn.h>
#include <assert.h>
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

half *LoadvggKernel(string name, int *&kernel_width, int *&kernel_height, int batch_size)
{
    float temp;
    *kernel_width = *kernel_height = 3;
    half *kernel = new half[*kernel_width * *kernel_height * batch_size];
    for (int i = 0; i < batch_size; i++)
    {
        ifstream kernel_data("dataset/vggdata/kernel/" + name);
        for (int j = i * (*kernel_width * *kernel_height); j < (i + 1) * (*kernel_width * *kernel_height); j++)
        {   
            kernel_data >> temp;
            kernel[j]=__float2half(temp);
        }
        kernel_data.close();
    }

    return kernel;
}

// half return type should be replaced with float

float *LoadKernel(string name, int *&kernel_width, int *&kernel_height, int batch_size, int index)
{
    float temp;
    // ifstream kernel_shape("/home/syt/conv_pool/conv_pool/dataset/kernel/kernel_shape/" + name);
    // for (int i = 0; i < 2; i++)
    // {
    //     kernel_shape >> *kernel_width >> *kernel_height;
    // }
    int shape[49] = {3,
                     1, 3, 1, 1, 3, 1, 1, 3, 1,
                     1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1,
                     1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1,
                     1, 3, 1, 1, 3, 1, 1, 3, 1};
    *kernel_width = *kernel_height = shape[index];
    float *kernel = new float[*kernel_width * *kernel_height * batch_size];
    for (int i = 0; i < batch_size; i++)
    {
        ifstream kernel_data("/home/syt/conv_pool/conv_pool/dataset/resnetdataset_all/kernel/" + name);
        for (int j = i * (*kernel_width * *kernel_height); j < (i + 1) * (*kernel_width * *kernel_height); j++) {
            kernel_data >> temp;
            kernel[j]=__float2half(temp);
       }
       kernel_data.close();
    }

    return kernel;
}

half *LoadVggConvWeigth(int *&fea_width, int *&fea_height, int batch_size, int index)
{
    float temp;
    string *name = new string[5];

    ifstream fea_name("dataset/vggdata/pooling_name");
    for (int i = 0; i < 5; i++)
    {
        fea_name >> name[i];
    }
    fea_name.close();

    int shape[5] = {224, 112, 56, 28, 14};

    *fea_width = *fea_height = shape[index];

    half *feature = new half[batch_size * *fea_width * *fea_height];

    for (int i = 0; i < batch_size; i++)
    {
        ifstream fea_data("dataset/vggdata/feature/" + name[index]);
        for (int j = i * (*fea_width * *fea_height); j < (i + 1) * (*fea_width * *fea_height); j++)
        {    
            fea_data >> temp;
            feature[j]=__float2half(temp);
        }
        fea_data.close();
    }

    return feature;
}

float *LoadConvWeight(int *&fea_width, int *&fea_height, int batch_size, int index)
{
    string *name = new string[49];

    ifstream fea_name("/home/syt/conv_pool/conv_pool/dataset/resnetdataset_all/feature_name.txt");
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
        ifstream fea_data("/home/syt/conv_pool/conv_pool/dataset/resnetdataset_all/feature/" + name[index]);
        for (int j = i * (*fea_width * *fea_height); j < (i + 1) * (*fea_width * *fea_height); j++) {
            fea_data >> feature[j];
        }    
        fea_data.close();
    }

    return feature;
}

float *LoadspConvWeigth(int *&fea_width, int *&fea_height, int batch_size, int index)
{
    string *name = new string[9];

    ifstream fea_name("/home/syt/conv_pool/conv_pool/sparsity/dataset/sparsity_name.txt");
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
        ifstream fea_data("/home/syt/conv_pool/conv_pool/sparsity/dataset/" + name[index]);
        for (int j = i * (*fea_width * *fea_height); j < (i + 1) * (*fea_width * *fea_height); j++)
            fea_data >> feature[j];
        fea_data.close();
    }

    return feature;
}

int main(int argc, char *argv[])
{
    int batch_size = atoi(argv[1]);
    // load kernel file name
    string kername_name[16];
    // ifstream open_kernel("/home/syt/conv_pool/conv_pool/dataset/resnetdataset_all/kernel_name.txt");
    // string kername_name[49];
    ifstream open_kernel("dataset/vggdata/kernel_name.txt");
    for (int i = 0; i < 16; i++)
    {
        open_kernel >> kername_name[i];
    }
    open_kernel.close();

    ofstream time_file(string("ECR/ECR/time_vgg/batchsize") + argv[1] + string(".txt"));
    // ofstream time_file(string("/home/syt/conv_pool/conv_pool/PECR/cudnn/time_wino_no/res_batchsize") + argv[1] + string(".txt"));

    for (int i = 0; i < 5; i++)
    {

        cudnnHandle_t cudnn;
        CUDNN_CALL(cudnnCreate(&cudnn));

        // input
        int *fea_width_ = new int;
        int *fea_height_ = new int;
        half *in_data;
        half *matrix;
        // matrix = LoadConvWeight(fea_width_, fea_height_, batch_size, i);
        matrix = LoadVggConvWeigth(fea_width_, fea_height_, batch_size, i);
        // matrix = LoadspConvWeigth(fea_width_, fea_height_, batch_size, i);

        const int in_n = batch_size;
        const int in_c = 1;
        const int in_h = *fea_height_;
        const int in_w = *fea_width_;

        cudnnTensorDescriptor_t in_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
            in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
            in_n, in_c, in_h, in_w));

        // filter
        // const int filt_k = 1;
        // const int filt_c = 1;
        // const int filt_h = 3;
        // const int filt_w = 3;

        // float kernel[9] = {1, 0, 1, 0, 1, 0, 1, 0, 0};
        half *filt_data;
        // float *in_data;

        int *kernel_width_ = new int;
        int *kernel_height_ = new int;
        half *kernel;
        // kernel = LoadKernel(kername_name[i], kernel_width_, kernel_height_, batch_size, i);
        kernel = LoadvggKernel(kername_name[0], kernel_width_, kernel_height_, batch_size);
        const int filt_k = 1;
        const int filt_c = 1;
        const int filt_h = *kernel_width_;
        const int filt_w = *kernel_height_;
        // const int file_size = filt_h * filt_w * filt_c;

        // convolution
        const int pad_h = 0;
        const int pad_w = 0;
        const int str_h = 1;
        const int str_w = 1;
        const int dil_h = 1;
        const int dil_w = 1;
        // output
        int out_n;
        int out_c;
        int out_h;
        int out_w;

        half *out_data;

        cudaEvent_t start, stop;
        float elapsedTime1 = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        CUDA_CALL(cudaMalloc(
            &in_data, in_n * in_c * in_h * in_w * sizeof(float)/2));

        cudnnFilterDescriptor_t filt_desc;
        CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
        CUDNN_CALL(cudnnSetFilter4dDescriptor(
            filt_desc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
            filt_k, filt_c, filt_h, filt_w));

        CUDA_CALL(cudaMalloc(
            &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)/2));

        cudnnConvolutionDescriptor_t conv_desc;
        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
        CUDNN_CALL(cudnnSetConvolution2dDescriptor(
            conv_desc,
            pad_h, pad_w, str_h, str_w, dil_h, dil_w,
            CUDNN_CONVOLUTION, CUDNN_DATA_HALF));

        cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);

        CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
            conv_desc, in_desc, filt_desc,
            &out_n, &out_c, &out_h, &out_w));

        cudnnTensorDescriptor_t out_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
            out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
            out_n, out_c, out_h, out_w));

        CUDA_CALL(cudaMalloc(
            &out_data, out_n * out_c * out_h * out_w * sizeof(float)/2));

        // algorithm
        cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        // = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        // = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
        // = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        //= CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        // = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
        //=CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;//= CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED ;

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

        half *con_result = (half *)malloc(out_n * out_c * out_h * out_w * sizeof(float)/2);
        // perform
        half alpha = __float2half(1.f);
        half beta = __float2half(0.f);

        cudaMemcpy(filt_data, kernel, filt_k * filt_c * filt_h * filt_w * sizeof(float)/2, cudaMemcpyHostToDevice);
        cudaMemcpy(in_data, matrix, in_n * in_c * in_h * in_w * sizeof(float)/2, cudaMemcpyHostToDevice);

        CUDNN_CALL(cudnnConvolutionForward(
            cudnn,
            &alpha, in_desc, in_data, filt_desc, filt_data,
            conv_desc, algo, ws_data, ws_size,
            &beta, out_desc, out_data));

        // results

        CUDA_CALL(cudaMemcpy(
            con_result, out_data,
            out_n * out_c * out_h * out_w * sizeof(float)/2,
            cudaMemcpyDeviceToHost));

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

        cudnnHandle_t cudnn_p;
        CUDNN_CALL(cudnnCreate(&cudnn_p));
        cudnnPoolingDescriptor_t pooling_desc;
        CUDNN_CALL(cudnnCreatePoolingDescriptor(&pooling_desc));
        cudnnSetPooling2dDescriptor(pooling_desc,            // descriptor handle
                                    CUDNN_POOLING_MAX,       // mode - max pooling
                                    CUDNN_NOT_PROPAGATE_NAN, // NaN propagation mode
                                    2,                       // window height
                                    2,                       // window width
                                    0,                       // vertical padding
                                    0,                       // horizontal padding
                                    2,                       // vertical stride
                                    2);
        cudnnTensorDescriptor_t in_p_desc;
        cudnnCreateTensorDescriptor(&in_p_desc);
        cudnnSetTensor4dDescriptor(in_p_desc,         // descriptor handle
                                   CUDNN_TENSOR_NCHW, // data format
                                   CUDNN_DATA_HALF,  // data type (precision)
                                   out_n,             // number of images
                                   out_c,             // number of channels
                                   out_h,             // data height
                                   out_w);

        cudnnTensorDescriptor_t out_p_desc;
        cudnnCreateTensorDescriptor(&out_p_desc);
        cudnnSetTensor4dDescriptor(out_p_desc,        // descriptor handle
                                   CUDNN_TENSOR_NCHW, // data format
                                   CUDNN_DATA_HALF,  // data type (precision)
                                   1,                 // number of images
                                   1,                 // number of channels
                                   out_h / 2,         // data height
                                   out_w / 2);
        half a = __float2half(1.0f);
        half be = 0.0f;

        half *gpu_in;
        half *gpu_out;

        cudaMalloc(&gpu_in, out_n * out_c * out_h * out_w * sizeof(float)/2);
        cudaMalloc(&gpu_out, (out_h / 2) * (out_h / 2) * sizeof(float)/2);
        cudaMemset(out_data, 0, (out_h / 2) * (out_h / 2) * sizeof(float)/2);
        // cudaMemset(out_data, 0, (out_h / 2) * (out_h / 2) * sizeof(float));

        cudaMemcpy(gpu_in, con_result, out_n * out_c * out_h * out_w * sizeof(float)/2, cudaMemcpyHostToDevice);

        cudnnPoolingForward(cudnn_p,      // cuDNN context handle
                            pooling_desc, // pooling descriptor handle
                            &a,           // alpha scaling factor
                            in_p_desc,    // input tensor descriptor
                            gpu_in,       // input data pointer to GPU memory
                            &be,          // beta scaling factor
                            out_p_desc,   // output tensor descriptor
                            gpu_out);

        int result_size = out_n * out_c * (out_h / 2) * (out_w / 2);
        half *result = new half[result_size];
        cudaMemcpy(result, gpu_out, result_size * sizeof(float)/2, cudaMemcpyDeviceToHost);

        cudnnDestroyTensorDescriptor(in_p_desc);
        cudnnDestroyTensorDescriptor(out_p_desc);
        cudnnDestroyPoolingDescriptor(pooling_desc);
        cudnnDestroy(cudnn_p);

        cudaFree(gpu_in);
        cudaFree(gpu_out);

        free(con_result);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime1, start, stop);
        // cout << elapsedTime1 << endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // 写入时间
        // ofstream time_file("time/time_batchsize1.txt", ios::app);
        time_file << elapsedTime1 << endl;
        // time_file.close();
        // 结果写入文件
        // string file_name = "/home/lfa/fsy/syt/conv_pool/PECR/cudnn/output/output_file" + to_string(i);
        // // string file_name = "/home/syt/conv_pool/conv_pool/PECR/cudnn/out_resnet_fast/output" + to_string(i);

        // ofstream output_file(file_name);
        // for (int i = 0; i < result_size; i++)
        // {
        //     output_file << result[i] << " ";
        //     if ((i + 1) % (out_w / 2) == 0)
        //         output_file << "\n";
        // }
        // output_file.close();
        free(result);
    }
    time_file.close();
    return 0;
}
