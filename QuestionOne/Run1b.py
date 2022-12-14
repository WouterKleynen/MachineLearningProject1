import numpy as np
from Tools import createCSVClusterFilesKernel
from KernelKMeansClustering import KernelKMeansClustering

def getEuclideanDistance(a,b):          # Calculates Euclidean distance
    return np.linalg.norm(a-b)

def gaussianKernel(point1, point2):    # Calculates Gaussian distance
    return np.exp(-((getEuclideanDistance(point1, point2)**2)/(2 * sigma**2)))

def runFirstIterationKernel(dataSetFilePath, K, kernel):                          # Only called for the first iteration
    algorithmValues = KernelKMeansClustering(dataSetFilePath, K, kernel)          # Create an instance of the class              
    algorithmValues.firstIteration()                                              # Run the firstIteration() function                   
    return algorithmValues

def runNewIterationKernel(algorithmValues, K):
    print(f"Current division of amount of ID's per cluster {algorithmValues.getClusterVectorSizesVector()}")
    previousLossFunctionValue = algorithmValues.calculateLossFunctionValue()
    createCSVClusterFilesKernel(K)
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
    amountOfIterations              = 1
    while (lossvalue != None): 
        print(f"Current algrotihm iteration = {amountOfIterations}")                                                             
        lossvalue = runNewIterationKernel(algorithmValues, K) 
        amountOfIterations += 1
    return lossvalue

kernel              = gaussianKernel                  # The Gaussian kernel is used 
sigma               = 1.7                             # Sigma is chosen so the output values of the function aren't really small or really big, but between 0 and 1
dataSetFilePath     = 'Dataset/InputData.csv'
#dataSetFilePath    = 'Dataset/subsetOfInputData.csv'                                                       
K                   = 8

improveUntilUnchanged(dataSetFilePath, K, gaussianKernel)