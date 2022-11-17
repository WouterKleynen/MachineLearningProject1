from NEW import NEW
import numpy as np

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
print(algorithmValues.clusterDictionary)
print(algorithmValues.getClusterVectorSizesVector())
algorithmValues.standardizeData()
algorithmValues.setKAccentValues()
algorithmValues.setKernelClusterDictionary()
print(algorithmValues.clusterDictionary)
print(algorithmValues.getClusterVectorSizesVector())
# # algorithmValues.setKAccentValues()
# # algorithmValues.setKernelClusterDictionary()



