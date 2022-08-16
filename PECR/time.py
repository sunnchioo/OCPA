import os

for i in range(2, 129, 2):
    os.system('./cudnn/cudnn '+str(i))
    os.system('./cudnn/cudnn '+str(i))
    os.system('./cudnn/cudnn '+str(i))
    os.system('./cudnn/cudnn '+str(i))
    os.system('./pecr/batchedPECR.out '+str(i))
    os.system('./pecr/batchedPECR.out '+str(i))
    os.system('./pecr/batchedPECR.out '+str(i))
    os.system('./pecr/batchedPECR.out '+str(i))
    print(i)