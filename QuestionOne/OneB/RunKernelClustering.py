from KernelKMeansClustering import KernelKMeansClustering
from Tools import createCSVClusterFilesKernel

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



dataSetFilePath     = 'Dataset/subsetOfInputData.csv'                                                       
improveUntilUnchanged(dataSetFilePath, 10, 1000)