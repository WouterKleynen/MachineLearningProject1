from K_MeansClusteringEuclidean import *

K = 10
dataSetFilePath = 'Dataset/EastWestAirlinesCluster.csv'

classInstance = K_MeansClusteringEuclidean(dataSetFilePath, K)
classInstance.getMaximaColumns()
classInstance.setCentroids()
classInstance.getDistances()
print(classInstance.centroidToPointsDistances)