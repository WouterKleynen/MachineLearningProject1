import math
import numpy as np
from KMeansClusteringKernel import *
from Tools import createCSVClusterFilesKernel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def standardize(column):                                                                
    mu = np.average(column)
    sigma = np.std(column)
    Z = (column - mu)/sigma
    return Z

def standardizeData(data):                                                                          
    standardizedMatrix = np.zeros((data.shape[0], data.shape[1]))
    numberOfColumns = data.shape[1]
    for i in range(0, numberOfColumns):
        standardizedMatrix[:, i] = standardize(data[:, i])                                        
    pd.DataFrame(standardizedMatrix).to_csv("Dataset\standardizedData.csv",index=False, header=False)
    return standardizedMatrix

def runFirstIterationKernel(dataSetFilePath, K, sigma):
    algorithmValues = KMeansClusteringKernel(dataSetFilePath, K, sigma)                              
    algorithmValues.firstIteration()                                                                 
    return algorithmValues

def runNewIterationKernel(algorithmValues, K):
    oldClusterVectorSizesVector = algorithmValues.getClusterVectorSizesVector()
    print(oldClusterVectorSizesVector)                       
    createCSVClusterFilesKernel(K)                                                                  
    algorithmValues.improveLossFunctionValueKernel()                                                
    newClusterVectorSizesVector = algorithmValues.getClusterVectorSizesVector()
    if (newClusterVectorSizesVector  == oldClusterVectorSizesVector):                                
        algorithmValues.fillClusterCSV()                                                            
        return None           
    return newClusterVectorSizesVector

def improveUntilUnchanged(dataSetFilePath, K, sigma):
    algorithmValues                 = runFirstIterationKernel(dataSetFilePath, K, sigma)                                          
    newClusterVectorSizesVector     = algorithmValues.getClusterVectorSizesVector()        
    while (newClusterVectorSizesVector != None):                                                              
        newClusterVectorSizesVector = runNewIterationKernel(algorithmValues, K) 
    return newClusterVectorSizesVector

dataSetFilePath     = 'Dataset/InputData.csv'                                                       
data                = pd.read_csv(dataSetFilePath).to_numpy()
standardizedData    = standardizeData(data)
dataWithoutIDMatrix = standardizedData[:, 1:]
stadardizedPath = "Dataset/standardizedData.csv"
testPath = "Dataset/testing.csv"
algorithmvalues = runFirstIterationKernel(testPath, 10, 10)

improveUntilUnchanged(stadardizedPath, 10, 10)