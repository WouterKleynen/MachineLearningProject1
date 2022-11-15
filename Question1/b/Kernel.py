import math
import numpy as np
from KMeansClustering import *
from Tools import createCSVClusterFilesKernel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def standardize(column):                                                                # standardizes a column
    mu = np.average(column)
    sigma = np.std(column)
    Z = (column - mu)/sigma
    return Z

def standardizeData(data):                                                              # standardizes the entire data
    standardizedMatrix = np.zeros((data.shape[0], data.shape[1]))
    numberOfColumns = data.shape[1]
    for i in range(0, numberOfColumns):
        standardizedMatrix[:, i] = standardize(data[:, i])                                        
    pd.DataFrame(standardizedMatrix).to_csv("Dataset\standardizedData.csv",index=False, header=False)
    return standardizedMatrix

def runFirstIterationKernel(dataSetFilePath, K, sigma):
    algorithmValues = KMeansClusteringKernel(dataSetFilePath, K, sigma)                              # Create an instance of the KMeansClusteringEuclidean class.
    algorithmValues.firstIteration()                                                    # Set the start Centroids and fill each cluster with its closest data points for the first run of the algorithm.
    return algorithmValues

def runNewIteration(previousLossFunctionvalue, algorithmValues, K, threshold):
    createCSVClusterFilesKernel(K)                                                                  # Create CSV file for each cluster
    print(f"current loss fuction value = {previousLossFunctionvalue}")
    algorithmValues.improveLossFunctionValueKernel()                                      # Update the centroids by using the improveLossFunction() function
    newLossFunctionValue = algorithmValues.calculateLossFunctionValueKernel()             # Determine the value of the loss function after the new centroid update
    if (previousLossFunctionvalue/newLossFunctionValue < threshold):                                # Since newLossFunctionValue <= previousLossFuncitonvalue we get a decreasing number, we stop when they're very close i.e. their fraction is very small
        algorithmValues.fillClusterCSV()                                            # Fill each cluster's CSV file with its datapoints
        print(f"Final loss function value for K = {K} is {newLossFunctionValue}")
        return None                                                                                 # Return None when the ratio is below the Treshold
    return newLossFunctionValue

    def 

dataSetFilePath     = 'Dataset/InputData.csv'                                                               # Set data File path to that of the assignment data sheet.
data                = pd.read_csv(dataSetFilePath).to_numpy()
standardizedData    = standardizeData(data)
dataWithoutIDMatrix = standardizedData[:, 1:]
# stadardizedPath = "Dataset/standardizedData.csv"
testPath = "Dataset/testing.csv"
runFirstIterationKernel(testPath, 10, 10)
