from operator import index
from time import time
import pandas as pd

path_cudnn = '/home/syt/conv_pool/conv_pool/PECR/cudnn/time_vgg_new/'
path_pecr = '/home/syt/conv_pool/conv_pool/PECR/pecr/time_vgg_new/'
# for i in range(2, 129, 2):
batch='batchsize32.txt'
time_cudnn = pd.read_csv(path_cudnn+batch,header=None,dtype=float)
time_pecr = pd.read_csv(path_pecr+batch,header=None,dtype=float)

# print(time_cudnn)

times=[]
for j in range(0, 5):
    # print(time_cudnn.iloc[j])
    times.append(time_cudnn[0][j]/time_pecr[0][j])

save = pd.DataFrame(times)
save.to_csv('/home/syt/conv_pool/conv_pool/PECR/pecr/times_vgg_new/times_'+batch, index=False, header=False, sep='\n')

