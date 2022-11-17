from KernelNonStandardizedMix import KernelNonStandardizedMix
import numpy as np
from Tools import createCSVClusterFilesKernel

# Gets Euclidean distance of 2 vectors 
def getEuclideanDistance(a,b):
    return np.linalg.norm(a-b)

sigma = 1.7
# Gets Gaussian distance of 2 vectors given a sigma
def gaussianKernel(point1, point2):
    return np.exp(-(getEuclideanDistance(point1, point2)/(2 * sigma**2)))

kernel = gaussianKernel
dataSetFilePath     = 'Dataset/subsetOfInputData.csv'                                                       
K = 10


def runFirstIterationKernel(dataSetFilePath, K, kernel):
    algorithmValues = KernelNonStandardizedMix(dataSetFilePath, K, kernel)                              
    algorithmValues.firstIteration()                                                                 
    return algorithmValues

def runNewIterationKernel(algorithmValues, K):
    previousLossFunctionValue = algorithmValues.calculateLossFunctionValue()
    print()
    createCSVClusterFilesKernel(K)
    print(algorithmValues.getClusterVectorSizesVector())                                                                  
    print(f"current loss fuction value = {previousLossFunctionValue}")
    algorithmValues.improveKernelLossFunctionValue()                                                
    newLossFunctionValue = algorithmValues.calculateLossFunctionValue()
    if (previousLossFunctionValue  == newLossFunctionValue): 
        print(algorithmValues.calculateLossFunctionValue())       
        print(algorithmValues.getClusterVectorSizesVector())                                                                                          
        algorithmValues.fillClusterCSV()                                                            
        return None           
    return newLossFunctionValue

def improveUntilUnchanged(dataSetFilePath, K, kernel):
    algorithmValues                 = runFirstIterationKernel(dataSetFilePath, K, kernel)                                          
    lossvalue                       = algorithmValues.calculateLossFunctionValue()        
    while (lossvalue != None):                                                              
        lossvalue = runNewIterationKernel(algorithmValues, K) 
    return lossvalue

improveUntilUnchanged(dataSetFilePath, K, gaussianKernel)