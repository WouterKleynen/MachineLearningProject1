import math
import numpy as np
from KMeansClustering import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def standardize(column):
    mu = np.average(column)
    sigma = np.std(column)
    Z = (column - mu)/sigma
    return Z

def standardizeData(data):
    standardizedMatrix = np.zeros((data.shape[0], data.shape[1]))
    numberOfColumns = data.shape[1]
    for i in range(0, numberOfColumns):
        standardizedMatrix[:, i] = standardize(data[:, i])                                        
    pd.DataFrame(standardizedMatrix).to_csv("Dataset\standardizedData.csv",index=False, header=False)
    return standardizedMatrix

def runFirstIteration(dataSetFilePath, K):
    currentAlgorithmIterationValues = KMeansClustering(dataSetFilePath, K)                         # Create an instance of the KMeansClusteringEuclidean class.
    currentAlgorithmIterationValues.firstIteration()                                                # Set the start Centroids and fill each cluster with its closest data points for the first run of the algorithm.
    return currentAlgorithmIterationValues


dataSetFilePath = 'Dataset/InputData.csv'                                                           # Set data File path to that of the assignment data sheet.
data = pd.read_csv(dataSetFilePath).to_numpy()
print(data)
standardizedData = standardizeData(data)
dataWithoutIDMatrix = standardizedData[:, 1:]
stadardizedPath = "Dataset/standardizedData.csv"
testPath = "Dataset/testing.csv"
runFirstIteration(testPath, 10)