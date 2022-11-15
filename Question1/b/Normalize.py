import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def standardize(column):
    mu = np.average(column)
    sigma = np.std(column)
    Z = (column - mu)/sigma
    return Z

def standardizeData(data):
    standardizedMatrix = np.zeros((data.shape[0], data.shape[1] - 1))
    numberOfColumns = data.shape[1]
    for i in range(0, numberOfColumns):
        standardizedMatrix[:, i-1] = standardize(data[:, i])                                        
    pd.DataFrame(standardizedMatrix).to_csv("Dataset\standardizedData.csv",index=False, header=False)

dataSetFilePath = 'Dataset/InputData.csv'                                                           # Set data File path to that of the assignment data sheet.
data = pd.read_csv(dataSetFilePath).to_numpy()
dataWithoutIDMatrix = data[:, 1:]
standardizeData(data)