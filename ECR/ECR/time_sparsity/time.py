import os
import pandas as pd

cudnn_path = '/home/syt/conv_pool/conv_pool/ECR/cudnn/time_sparsity_new/batchsize32.txt'
ecr_path = '/home/syt/conv_pool/conv_pool/ECR/ECR/time_sparsity_new/batchsize32.txt'
time1 = [0 for i in range(9)]
time2 = [0 for i in range(9)]
for i in range(1000):
    os.system('/home/syt/conv_pool/conv_pool/ECR/cudnn/cudnn 32')
    os.system('/home/syt/conv_pool/conv_pool/ECR/ECR/batchedECR.out 32')
    data1=pd.read_csv(cudnn_path,header=None,dtype=float)
    data2=pd.read_csv(ecr_path,header=None,dtype=float)

    for j in range(0, 9):
        time1[j]=time1[j]+data1[0][j]
        time2[j]=time2[j]+data2[0][j]

for n in range(0,9):
    time1[n]=time1[n]/1000
    time2[n]=time2[n]/1000
    print(time1[n]/time2[n])

