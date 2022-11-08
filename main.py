from K_MeansClusteringEuclidean import *

K = 10
dataSetFilePath = 'Dataset/EastWestAirlinesCluster.csv'

classInstance = K_MeansClusteringEuclidean(dataSetFilePath, K)
classInstance.getMaximaOfColumns()
classInstance.setStartCentroids()
print(classInstance.lossFunction())

classInstance.getDistanceOfPointsToCentroids()
classInstance.setClusterDictionary()
classInstance.updateCentroids()
print(classInstance.lossFunction())

classInstance.getDistanceOfPointsToCentroids()
classInstance.setClusterDictionary()
classInstance.updateCentroids()
print(classInstance.lossFunction())

classInstance.getDistanceOfPointsToCentroids()
classInstance.setClusterDictionary()
classInstance.updateCentroids()
print(classInstance.lossFunction())



