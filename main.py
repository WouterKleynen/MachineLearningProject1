from K_MeansClusteringEuclidean import *

K = 10
dataSetFilePath = 'Dataset/EastWestAirlinesCluster.csv'

classInstance = K_MeansClusteringEuclidean(dataSetFilePath, K)
classInstance.getMaximaOfColumns()
classInstance.setStartCentroids()
print(classInstance.lossFunction())

for i in range(50):
    classInstance.getDistanceOfPointsToCentroids()
    classInstance.setClusterDictionary()
    classInstance.updateCentroids()
    print(classInstance.lossFunction())