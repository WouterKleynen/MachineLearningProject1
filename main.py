from K_MeansClusteringEuclidean import *

K = 10
dataSetFilePath = 'Dataset/EastWestAirlinesCluster.csv'

classInstance = K_MeansClusteringEuclidean(dataSetFilePath, K)
classInstance.getMaximaOfColumns()
classInstance.setStartCentroids()
classInstance.getDistanceOfPointsToCentroids()
classInstance.updateClusterDictionary()
classInstance.getSizeOfClusters()
print(classInstance.clusterSizes)
