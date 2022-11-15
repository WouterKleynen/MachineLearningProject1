import math
import numpy as np
from KMeansClusteringEuclidean import *

dataSetFilePath = 'Dataset/testing.csv'                                                           # Set data File path to that of the assignment data sheet.

def runFirstIteration(K):
    currentAlgorithmIterationValues = KMeansClusteringEuclidean(dataSetFilePath, K)                 # Create an instance of the KMeansClusteringEuclidean class.
    currentAlgorithmIterationValues.firstIteration()                                                # Set the start Centroids and fill each cluster with its closest data points for the first run of the algorithm.
    return currentAlgorithmIterationValues

    
K = 10
values = runFirstIteration(K)
sigma = 1000 
point1 = np.array([28143,0,1,1,1,174,1,0,0,7000,0])
point2 = np.array([97752,0,4,1,1,43300,26,2077,4,6935,1])

# print(getGaussianDistance(point1, point2, sigma))
# print(getGaussianDistance(point1, point2, sigma))


# print(values.clusterDictionary)
print(values.sumOfGaussianDistanceWithAllPoints(1, sigma))
