from RunAlgorithm import improveUntilTresholdReached
import pandas as pd
from AlgorithmClass import *
from RunAlgorithm import *
from Tools import createCSVClusterFiles


def improveUntilTresholdReachedWithCSV(K, treshold, printIntermediateLossFunctionValues=False):
    # Create CSV file for each cluster
    createCSVClusterFiles(K)
    # Update to first Iteration (this differs from other iteration since it has to construct start centroids)
    currentAlgorithmIterationValues = runFirstIteration(K)
    # Calculate the start loss function value after the first iteration
    startLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()
    # set previousLossFuncitonvalue to startLossFunctionValue so they can be compared in the for loop
    previousLossFunctionvalue = startLossFunctionValue
    currentAlgorithmIterationValues.fillClusterCSV()
    # # loop from 0 untill the iteration that the treshold is reached: when previousLossFunctionvalue == None
    # while (previousLossFunctionvalue != None):
    #     # update each previous loss function value with a new improved one
    #     previousLossFunctionvalue = runNewIterationWithCSV(previousLossFunctionvalue, currentAlgorithmIterationValues, K, treshold, printIntermediateLossFunctionValues)

# Is called in every iteration to decrease the Loss Function. If the intermediate loss function values need to be printed, set printIntermediateLossFunctionValues to true
def runNewIterationWithCSV(previousLossFunctionvalue, currentAlgorithmIterationValues, K, threshold, printIntermediateLossFunctionValues = False):
    if printIntermediateLossFunctionValues == True:
        print(f"current loss fuction value = {previousLossFunctionvalue}")
    # Update the centroids by using the improveLossFunction() function
    currentAlgorithmIterationValues.improveLossFunctionValue()
    # Determine the value of the loss function after the new centroid update
    newLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()
    # Since newLossFunctionValue <= previousLossFuncitonvalue we get a decreasing number, we stop when they're very close i.e. their fraction is very small
    if (previousLossFunctionvalue/newLossFunctionValue < threshold):
        print(f"Final loss function value for K = {K} is {newLossFunctionValue}")
        # Fill each cluster's CSV file with its datapoints
        currentAlgorithmIterationValues.fillClusterCSV()
        # Return None when the ratio is below the Treshold
        return None
    # update the loss function value to be able to compare the new value to the old value
    return newLossFunctionValue

improveUntilTresholdReachedWithCSV(10, 1.000_01, True)