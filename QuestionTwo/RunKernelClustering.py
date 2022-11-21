import numpy as np
from Tools import createCSVClusterFilesKernel
from Run1b import KernelKMeansClustering

def getEuclideanDistance(a,b):
    return np.linalg.norm(a-b)

sigma = 1.7
def gaussianKernel(point1, point2):
    return np.exp(-(getEuclideanDistance(point1, point2)/(2 * sigma**2)))

kernel              = gaussianKernel
dataSetFilePath     = 'Dataset/InputData.csv'                                                       
K                   = 10

def runFirstIterationKernel(dataSetFilePath, K, kernel):
    algorithmValues = KernelKMeansClustering(dataSetFilePath, K, kernel)                              
    algorithmValues.firstIteration()                                                                 
    return algorithmValues

def runNewIterationKernel(algorithmValues, K):
    previousLossFunctionValue = algorithmValues.calculateLossFunctionValue()
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