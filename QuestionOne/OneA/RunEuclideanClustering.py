import pandas as pd
from EuclideanKMeansClustering import EuclideanKMeansClustering

# This file contains the functions that are used to iteratively call the KMeansClusteringEuclidean() class functions
# to run the K Means Euclidean Clustering algorithm until the given the treshold is reached

# Is called to run the first iteration. The first iteration differs from other iteration since it has to construct start centroids.
def runFirstIteration(dataSetFilePath, K):
    currentAlgorithmIterationValues = EuclideanKMeansClustering(dataSetFilePath, K)                 # Create an instance of the KMeansClusteringEuclidean class.
    currentAlgorithmIterationValues.firstIteration()                                                # Set the start Centroids and fill each cluster with its closest data points for the first run of the algorithm.
    return currentAlgorithmIterationValues

# Is called in every iteration to decrease the Loss Function. If the intermediate loss function values need to be printed, set printIntermediateLossFunctionValues to true
def runNewIteration(previousLossFunctionvalue, currentAlgorithmIterationValues, K):
    currentAlgorithmIterationValues.createCSVClusterFiles(K) 
    print(currentAlgorithmIterationValues.getClusterVectorSizesVector())                                                                  
    # Create CSV file for each cluster
    print(f"current loss fuction value = {previousLossFunctionvalue}")
    currentAlgorithmIterationValues.improveLossFunctionValue()                                      # Update the centroids by using the improveLossFunction() function
    newLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()             # Determine the value of the loss function after the new centroid update
    if (previousLossFunctionvalue == newLossFunctionValue):                                # Since newLossFunctionValue <= previousLossFuncitonvalue we get a decreasing number, we stop when they're very close i.e. their fraction is very small
        currentAlgorithmIterationValues.fillClusterCSV()                                            # Fill each cluster's CSV file with its datapoints
        print(currentAlgorithmIterationValues.getClusterVectorSizesVector())
        print(f"Final loss function value for K = {K} is {newLossFunctionValue}")
        return None                                                                                 # Return None when the ratio is below the Treshold
    return newLossFunctionValue

# Runs the K Means Euclidean Clustering algorithm for a given K unt
def improveUntilTresholdReached(dataSetFilePath, K):
    currentAlgorithmIterationValues = runFirstIteration(dataSetFilePath, K)                         # Update to first Iteration (this differs from other iteration since it has to construct start centroids)
    lossFunctionvalue = currentAlgorithmIterationValues.calculateLossFunctionValue()                # Calculate the start loss function value after the first iteration
    while (lossFunctionvalue != None):                                                              # loop from 0 untill the iteration that the treshold is reached: when previousLossFunctionvalue == None
        lossFunctionvalue = runNewIteration(lossFunctionvalue, currentAlgorithmIterationValues, K)  # update each previous loss function value with a new improved one
    return lossFunctionvalue

dataSetFilePath = 'Dataset/InputData.csv'   # Set data File path to that of the assignment data sheet.
testData = "Dataset/subsetOfInputData.csv"

improveUntilTresholdReached(dataSetFilePath, 10)