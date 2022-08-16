import os

for i in range(0, 100):
    os.system('CUDA_VISIBLE_DEVICES=1 ./ECR/batchedECR.out 32 '+str(i))
    print(i)