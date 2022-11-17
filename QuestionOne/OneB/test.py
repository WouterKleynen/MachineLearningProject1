from NEW import NEW
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
algorithmValues = NEW(dataSetFilePath, K, kernel)                              
algorithmValues.firstIteration()
algorithmValues.standardizeData()

for runIndex in range(5):
    print(algorithmValues.getClusterVectorSizesVector())
    print(algorithmValues.calculateLossFunctionValue())
    #print(algorithmValues.centroidsMatrix)
    
    algorithmValues.setKAccentValues()
    algorithmValues.setKernelClusterDictionary()
    algorithmValues.setKernelCentroids

# def runFirstIterationKernel(dataSetFilePath, K, kernel):
#     algorithmValues = NEW(dataSetFilePath, K, kernel)                              
#     algorithmValues.firstIteration()                                                                 
#     return algorithmValues


# def runNewIterationKernel(algorithmValues, K):
#     previousLossFunctionValue = algorithmValues.calculateLossFunctionValue()
#     createCSVClusterFilesKernel(K)
#     print(algorithmValues.getClusterVectorSizesVector())                                                                  
#     print(f"current loss fuction value = {previousLossFunctionValue}")
#     algorithmValues.improveLossFunctionValueKernel()                                                
#     newLossFunctionValue = algorithmValues.calculateLossFunctionValue()
#     if (newLossFunctionValue  == previousLossFunctionValue): 
#         print(algorithmValues.calculateLossFunctionValue())       
#         print(algorithmValues.getClusterVectorSizesVector())                                                                                          
#         algorithmValues.fillClusterCSV()                                                            
#         return None           
#     return newLossFunctionValue
