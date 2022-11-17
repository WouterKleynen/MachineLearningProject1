import pandas as pd
import numpy as np
import math
from scipy.linalg import eigh

#variables
data = pd.read_csv('Dataset/InputData.csv').to_numpy()
K = 10
sample_size = data.shape[0]

#define the kernel function we will use
def Kernel(a,b):
    func = np.exp(-(np.linalg.norm(a - b)/(2 * (1.7)*2))*2)
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

#create EVM and EVV
#EVextors is a matrix where the columns are the eigenvectors of matrix L
#EValues is a vector with the corresponding eigenvalues (not necessary for this assignment but might be useful later)
EValues, EVectors = eigh(Laplacian, eigvals= (0,K-1))

#run K-means with the 10 vectors (columns) of EVectors as the centroids to find the clustering corresponding
#to the 10 smallest eigenvectors