from K_MeansClusteringEuclidean import *

K = 10
dataSetFilePath = 'Dataset/EastWestAirlinesCluster.csv'


k_MeansClusteringEuclidean = setStartCentroids(K_MeansClusteringEuclidean(dataSetFilePath, K))
startLossFunctionValue = k_MeansClusteringEuclidean.getLossFunctionValue()
previousLossFuncitonvalue = startLossFunctionValue

for i in range(50):
    print(previousLossFuncitonvalue)
    improveLossFunction(k_MeansClusteringEuclidean)
    newLossFunctionValue = k_MeansClusteringEuclidean.getLossFunctionValue()
    if (previousLossFuncitonvalue - newLossFunctionValue < 10):
        print(f"Final lost function value = {newLossFunctionValue}")
        break
    previousLossFuncitonvalue = newLossFunctionValue