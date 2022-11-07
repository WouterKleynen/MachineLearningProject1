import K_MeansClusteringEuclidean


classInstance = K_MeansClusteringEuclidean('Dataset/EastWestAirlinesCluster.csv', 10)
classInstance.getMaximaColumns()
classInstance.updateCentroids()
classInstance.getDistances()
print(classInstance.distances)