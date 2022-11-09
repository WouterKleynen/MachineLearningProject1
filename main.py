# import from K_MeansClusteringEuclidean.py where all the used functions are located.
from KMeansClustering.KMeansClusteringEuclidean import *
# Set the amount of Cluster (K) to 10.
K = 10
# Set data File path to that of the assignment data sheet.
dataSetFilePath = 'Dataset/EastWestAirlinesCluster.csv'
# Create an instance of the KMeansClusteringEuclidean class.
k_MeansClusteringEuclidean = KMeansClusteringEuclidean(dataSetFilePath, K)
# Set the start Centroids and fill each cluster with its closest data points for the first run of the algorithm.
k_MeansClusteringEuclidean.firstIteration()
# Calculate the start loss function value after the first iteration
startLossFunctionValue = k_MeansClusteringEuclidean.calculateLossFunctionValue()
# set previousLossFuncitonvalue to startLossFunctionValue so they can be compared in the for loop
previousLossFuncitonvalue = startLossFunctionValue

# loop from 0 to a very higher number so the centroids can be updated in each loop until the stopping criterium is reached
for i in range(10_000_000):
    print(previousLossFuncitonvalue)
    # Update the centroids by using the improveLossFunction() function
    k_MeansClusteringEuclidean.improveLossFunctionValue()
    # Determine the value of the loss function after the new centroid update
    newLossFunctionValue = k_MeansClusteringEuclidean.calculateLossFunctionValue()
    # Since newLossFunctionValue <= previousLossFuncitonvalue we get a decreasing number, we stop when they're very close i.e. their fraction is very small
    if (previousLossFuncitonvalue/newLossFunctionValue < 1.000_001):
        print(f"Final loss function value = {newLossFunctionValue}")
        # Quit the for loop when this condition is met
        break
    # update the loss function value to be able to compare the new value to the old
    previousLossFuncitonvalue = newLossFunctionValue