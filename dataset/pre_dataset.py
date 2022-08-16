import numpy as np
import os
import pandas as pd

# path = 'feature/feature_dataset'
path = 'kernel/kernel_dataset'
file_list = os.listdir(path)

file_list.sort()
# np.savetxt('feature/feature_name.txt', file_list, fmt='%s')
np.savetxt('kernel/kernel_name.txt', file_list, fmt='%s')

def feature():
    for file in file_list:
        # data = np.loadtxt(path+'/'+file, delimiter=' ')
        fea_name = file.split('.')[0]
        fea_shape = fea_name.split('_')[-2:]
        print(file)
        print(fea_shape)
        save = [int(fea_shape[0]), int(fea_shape[1])]
        print(save)
        np.savetxt('kernel/kernel_shape/'+file, save, fmt='%d')
        # break

def kernel():
    for file in file_list:
        data = pd.read_csv(path+'/'+file, delimiter=' ', header=None)
        save = data.shape
        print(save)
        np.savetxt('kernel/kernel_shape/'+file, save, fmt='%d')
        
# kernel()

