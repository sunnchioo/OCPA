import pandas as pd

path_cudnn = 'ECR/cudnn/time_vgg/'
path_ecr = 'ECR/ECR/time_vgg/'
batch='batchsize32.txt'

time_cudnn = pd.read_csv(path_cudnn+batch,header=None,dtype=float)
time_ecr = pd.read_csv(path_ecr+batch,header=None,dtype=float)


times=[]
for j in range(0, 16):
    times.append(time_cudnn[0][j]/time_ecr[0][j])

save = pd.DataFrame(times)
save.to_csv('ECR/times/'+batch, index=False, header=False, sep='\n')

