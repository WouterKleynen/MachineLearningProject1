import pandas as pd
from AlgorithmClass import KMeansClusteringEuclidean
from Tools import createCSVClusterFiles

# This file contains the functions that are used to iteratively call the KMeansClusteringEuclidean() class functions
# to run the K Means Euclidean Clustering algorithm until the given the treshold is reached

dataSetFilePath = 'Dataset/InputData.csv'                                                           # Set data File path to that of the assignment data sheet.

# Is called to run the first iteration. The first iteration differs from other iteration since it has to construct start centroids.
def runFirstIteration(K):
    currentAlgorithmIterationValues = KMeansClusteringEuclidean(dataSetFilePath, K)                 # Create an instance of the KMeansClusteringEuclidean class.
    currentAlgorithmIterationValues.firstIteration()                                                # Set the start Centroids and fill each cluster with its closest data points for the first run of the algorithm.
    return currentAlgorithmIterationValues

# Is called in every iteration to decrease the Loss Function. If the intermediate loss function values need to be printed, set printIntermediateLossFunctionValues to true
def runNewIteration(previousLossFunctionvalue, currentAlgorithmIterationValues, K, threshold, printIntermediateLossFunctionValues = False):
    createCSVClusterFiles(K)                                                                        # Create CSV file for each cluster
    if printIntermediateLossFunctionValues == True:                                                 # If the optional value printIntermediateLossFunctionValues is set to True then print
        print(f"current loss fuction value = {previousLossFunctionvalue}")
    currentAlgorithmIterationValues.improveLossFunctionValue()                                      # Update the centroids by using the improveLossFunction() function
    newLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()             # Determine the value of the loss function after the new centroid update
    if (previousLossFunctionvalue/newLossFunctionValue < threshold):                                # Since newLossFunctionValue <= previousLossFuncitonvalue we get a decreasing number, we stop when they're very close i.e. their fraction is very small
        currentAlgorithmIterationValues.fillClusterCSV()                                            # Fill each cluster's CSV file with its datapoints
        print(f"Final loss function value for K = {K} is {newLossFunctionValue}")
        return None                                                                                 # Return None when the ratio is below the Treshold
    return newLossFunctionValue                                                                     # update the loss function value to be able to compare the new value to the old value

# Runs the K Means Euclidean Clustering algorithm for a given K until the treshold is reached.
def improveUntilTresholdReached(K, treshold, printIntermediateLossFunctionValues=False):
    currentAlgorithmIterationValues = runFirstIteration(K)                                          # Update to first Iteration (this differs from other iteration since it has to construct start centroids)
    startLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()           # Calculate the start loss function value after the first iteration
    lossFunctionvalue = startLossFunctionValue                                                      # set lossFunctionvalue to startLossFunctionValue so they can be compared in the for loop
    while (lossFunctionvalue != None):                                                              # loop from 0 untill the iteration that the treshold is reached: when previousLossFunctionvalue == None
        lossFunctionvalue = runNewIteration(lossFunctionvalue, currentAlgorithmIterationValues, K, treshold, printIntermediateLossFunctionValues) # update each previous loss function value with a new improved one
    return lossFunctionvalue

improveUntilTresholdReached(10, 1.00000001, True)