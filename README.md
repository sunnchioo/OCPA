# OCPA
This repository is the implementation of Accelerating Convolutional Neural Network by Exploiting Sparsity on GPUs

## Requirements
+ Ubuntu 18.04
+ cuda == 10.2
+ cuDNN == 7.6

## Experiments
+ Vgg-16

running ECR for convolution layer

```
cd OCPA/ECR/ECR/time_vgg
nvcc batchedECR.cu -o batchedECR.out
./batchedECR.out 32
```

running PECR for convolution+pooling layer

```
cd OCPA/PECR/pecr/time_vgg
nvcc batchedPECR.cu -o batchedPECR.out
./batchedPECR.out 32
```

running cudnn(using tensor core) for convolution layer

```
cd OCPA/ECR/cudnn/time_vgg
make
./cudnn 32
```

running cudnn(using tensor core) for convolution+pooling layer

```
cd OCPA/PECR/cudnn/time_vgg
make
./cudnn 32
```

+ Resnet-50

running ECR for convolution layer

```
cd OCPA/ECR/ECR/time_resnet
nvcc batchedECR.cu -o batchedECR.out
./batchedECR.out 32
```

running PECR for convolution+pooling layer

```
cd OCPA/PECR/pecr/time_resnet
nvcc batchedPECR.cu -o batchedPECR.out
./batchedPECR.out 32
```

running cudnn(using tensor core) for convolution layer

```
cd OCPA/ECR/cudnn/time_resnet
make
./cudnn 32
```

running cudnn(using tensor core) for convolution+pooling layer

```
cd OCPA/PECR/pecr/time_resnet
make
./cudnn 32
```

We can get the running time of other algorithms by analogy with the above methods using cuDNN.

## Expected Result

The Vgg-16 and Resnet-50 speedup effects can be obtained by running programs under the folder `OCPA/speedup`. As shown in the following figure：

<!--![Figure 9](https://github.com/sunnchioo/OCPA/blob/main/speedup/figure9.png "speedup")-->
<img src="https://github.com/sunnchioo/OCPA/blob/main/speedup/figure9.png" width="50%">
