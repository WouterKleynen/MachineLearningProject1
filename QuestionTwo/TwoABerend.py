import pandas as pd
import numpy as np
import math
from scipy.linalg import eigh
import scipy.linalg

#variables
data = pd.read_csv('Dataset/InputData.csv').to_numpy()
K = 8
sample_size = data.shape[0]
data.shape
#define the kernel function we will use
def Kernel(a,b):
    func = np.exp(-(np.linalg.norm(a - b)**2/(2*((1.7)**2))))
    return func

#create W with zeroes
W = np.zeros((sample_size,sample_size))

#fill W with kernal distances
for i in range(sample_size):
    for j in range(sample_size):
        W[i,j] = Kernel(data[i,1:], data[j,1:])

#create D with zeroes
D = np.zeros((sample_size,sample_size))

#fill diagonal of D with sum of each column in W
for i in range(sample_size):
    D[i,i] = sum(W[i,:])

#create the graph Laplacian matrix
Laplacian = D - W
#create csv file of laplacian for use in other files
pd.DataFrame(Laplacian).to_csv("Dataset/Laplacian.csv", index = None)


#create EVM and EVV
#EVextors is a matrix where the columns are the eigenvectors of matrix L
#EValues is a vector with the corresponding eigenvalues (not necessary for this assignment but might be useful later)
EValues, EVectors = eigh(Laplacian, eigvals= (0,K-1))


print(EValues, EVectors)

#run K-means with the 10 vectors (columns) of EVectors as the centroids to find the clustering corresponding
#to the 10 smallest eigenvectors