# import from K_MeansClusteringEuclidean.py where all the used functions are located.
from KMeansClustering.KMeansClusteringEuclidean import *

# Set the amount of Cluster (K) to 10.
K = 10
# Set data File path to that of the assignment data sheet.
dataSetFilePath = 'Dataset/EastWestAirlinesCluster.csv'

# Create an instance of the KMeansClusteringEuclidean class.
k_MeansClusteringEuclidean = KMeansClusteringEuclidean(dataSetFilePath, K)

# Set the start Centroids and fill each cluster with it's closest data points for the first run of the algorithm.
k_MeansClusteringEuclidean = firstRun(k_MeansClusteringEuclidean)
startLossFunctionValue = k_MeansClusteringEuclidean.calculateLossFunctionValue()
previousLossFuncitonvalue = startLossFunctionValue

for i in range(50):
    print(previousLossFuncitonvalue)
    improveLossFunction(k_MeansClusteringEuclidean)
    newLossFunctionValue = k_MeansClusteringEuclidean.calculateLossFunctionValue()
    if (previousLossFuncitonvalue - newLossFunctionValue < 10):
        print(f"Final lost function value = {newLossFunctionValue}")
        break
    previousLossFuncitonvalue = newLossFunctionValue