from K_MeansClusteringEuclidean import *

classInstance = K_MeansClusteringEuclidean('Dataset/EastWestAirlinesCluster.csv', 10)
classInstance.getMaximaColumns()
classInstance.updateCentroids()
classInstance.getDistances()
print(classInstance.centroidToPointsDistances)