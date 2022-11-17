from KernelKMeansClustering import KernelKMeansClustering
from Tools import createCSVClusterFilesKernel
import numpy as np
import pandas as pd
import math

def runFirstIterationKernel(dataSetFilePath, K, kernel):
    algorithmValues = KernelKMeansClustering(dataSetFilePath, K, kernel)                              
    algorithmValues.firstIteration()                                                                 
    return algorithmValues

def runNewIterationKernel(algorithmValues, K):
    previousLossFunctionValue = algorithmValues.calculateLossFunctionValue()
    createCSVClusterFilesKernel(K)
    print(algorithmValues.getClusterVectorSizesVector())                                                                  
    print(f"current loss fuction value = {previousLossFunctionValue}")
    algorithmValues.improveLossFunctionValueKernel()                                                
    newLossFunctionValue = algorithmValues.calculateLossFunctionValue()
    if (newLossFunctionValue  == previousLossFunctionValue): 
        print(algorithmValues.calculateLossFunctionValue())       
        print(algorithmValues.getClusterVectorSizesVector())                                                                                          
        algorithmValues.fillClusterCSV()                                                            
        return None           
    return newLossFunctionValue


def sigmaOfStandardizedData(matrix):
    return np.std(matrix)

path = 'Dataset/standardizedData.csv'
matrix = pd.read_csv(path).to_numpy()
print(sigmaOfStandardizedData(matrix))


def improveUntilUnchanged(dataSetFilePath, K, kernel):
    algorithmValues                 = runFirstIterationKernel(dataSetFilePath, K, kernel)                                          
    newClusterVectorSizesVector     = algorithmValues.getClusterVectorSizesVector()        
    while (newClusterVectorSizesVector != None):                                                              
        newClusterVectorSizesVector = runNewIterationKernel(algorithmValues, K) 
    return newClusterVectorSizesVector

# Gets Euclidean distance of 2 vectors 
def getEuclideanDistance(a,b):
    return np.linalg.norm(a-b)

sigma = 1.7
# Gets Gaussian distance of 2 vectors given a sigma
def gaussianKernel(point1, point2):
    return np.exp(-(getEuclideanDistance(point1, point2)/(2 * sigma**2))**2)

kernel = gaussianKernel
dataSetFilePath     = 'Dataset/subsetOfInputData.csv'                                                       

improveUntilUnchanged(dataSetFilePath, 10, kernel)


3315787.509399832
5906155.188968996
