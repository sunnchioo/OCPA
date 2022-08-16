from numpy import dtype
import scipy as spy
import scipy.sparse as sp
import numpy as np

n=28
m=28
density=0.9
matrixformat='coo'
B=sp.rand(m,n,density=density,format=matrixformat,dtype=None)
# print(B)
A = B.todense()
print(A)
np.savetxt('./dataset/01.txt',A,fmt='%f')