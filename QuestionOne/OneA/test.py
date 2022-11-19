from EuclideanKMeansClustering import EuclideanKMeansClustering

dataSetFilePath     = 'Dataset/subsetOfInputData.csv'                                                       

K = 10
algorithmValues = EuclideanKMeansClustering(dataSetFilePath, K)                              
algorithmValues.firstIteration()

for runIndex in range(5):
    print(algorithmValues.getClusterVectorSizesVector())
    print(algorithmValues.calculateLossFunctionValue())
    print(algorithmValues.centroidsMatrix)
    algorithmValues.setDistanceOfPointsToCentroidsMatrix()
    algorithmValues.setClusterDictionary()
    algorithmValues.setCentroids()
