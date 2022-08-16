import pandas as pd

path = '/home/syt/conv_pool/conv_pool/ECR/speedup_resnet/'

time = pd.read_csv(path+'times_batchsize'+str(128)+'.txt',header=None,dtype=float)

print(time.sum()/32)