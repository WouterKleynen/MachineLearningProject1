from KernelKMeansClustering import KernelKMeansClustering
from Tools import createCSVClusterFilesKernel
import numpy as np
import math

def runFirstIterationKernel(dataSetFilePath, K, sigma):
    algorithmValues = KernelKMeansClustering(dataSetFilePath, K, sigma)                              
    algorithmValues.firstIteration()                                                                 
    return algorithmValues

def runNewIterationKernel(algorithmValues, K):
    oldClusterVectorSizesVector = algorithmValues.getClusterVectorSizesVector()
    print(oldClusterVectorSizesVector)                       
    createCSVClusterFilesKernel(K)                                                                  
    algorithmValues.improveLossFunctionValueKernel()                                                
    newClusterVectorSizesVector = algorithmValues.getClusterVectorSizesVector()
    if (newClusterVectorSizesVector  == oldClusterVectorSizesVector): 
        algorithmValues.setCentroids()
        print(algorithmValues.calculateLossFunctionValue())                               
        algorithmValues.fillClusterCSV()                                                            
        return None           
    return newClusterVectorSizesVector

def improveUntilUnchanged(dataSetFilePath, K, sigma):
    algorithmValues                 = runFirstIterationKernel(dataSetFilePath, K, sigma)                                          
    newClusterVectorSizesVector     = algorithmValues.getClusterVectorSizesVector()        
    while (newClusterVectorSizesVector != None):                                                              
        newClusterVectorSizesVector = runNewIterationKernel(algorithmValues, K) 
    return newClusterVectorSizesVector

# Gets Euclidean distance of 2 vectors 
def getEuclideanDistance(a,b):
    return np.linalg.norm(a-b)

sigma = 10
# Gets Gaussian distance of 2 vectors given a sigma
def gaussianKernel(point1, point2):
    return math.exp(-(getEuclideanDistance(point1, point2)/(2 * sigma**2))**2)

kernel = gaussianKernel
dataSetFilePath     = 'Dataset/subsetOfInputData.csv'                                                       
improveUntilUnchanged(dataSetFilePath, 10, kernel)