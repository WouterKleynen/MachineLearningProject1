import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)

dataSetFilePath = 'Dataset/InputData.csv'
data    = pd.read_csv(dataSetFilePath).to_numpy()
testData = data[:, 1:]
numberOfColumns = testData.shape[1]

def getAverageOfColumnsData(testData):
    return np.average(testData, axis=0)
    
def getStandardDeviation(testData):
    return np.std(testData, axis=0)

def getStdComparedToMuVector(testData):
    stdComparedToMuVector = []
    stdVector = getStandardDeviation(testData)
    muVector = getAverageOfColumnsData(testData)
    for index in range(0, len(muVector)):
        stdComparedToMuVector.append(muVector[index]/stdVector[index])
    return stdComparedToMuVector
    
print(getStdComparedToMuVector(testData))
    

    
