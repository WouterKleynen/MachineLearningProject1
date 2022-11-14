# import from K_MeansClusteringEuclidean.py the class KMeansClusteringEuclidean
from AlgorithmClass import KMeansClusteringEuclidean

# Set data File path to that of the assignment data sheet.
dataSetFilePath = 'Dataset/EastWestAirlinesCluster.csv'
#########################################################################################################
#  Functions used in main.py
#########################################################################################################

# Sets the first centroids by means of the maxima of the data columns
def firstIterationSteps(self):
    self.getMaximaOfColumns()
    self.setClusterDictionary()
    self.setStartCentroids()

# Is called in every loop to decrease the Loss function Value by resetting the centroids in a better wat
def improveLossFunctionValue(self):
    self.setDistanceOfPointsToCentroidsMatrix()
    self.setClusterDictionary()
    self.setCentroids()

# The first iteration differs from other iteration since it has to construct start centroids
def runFirstIteration(K):
    # Create an instance of the KMeansClusteringEuclidean class.
    currentAlgorithmIterationValues = KMeansClusteringEuclidean(dataSetFilePath, K)
    # Set the start Centroids and fill each cluster with its closest data points for the first run of the algorithm.
    currentAlgorithmIterationValues.firstIteration()
    return currentAlgorithmIterationValues

# Is called in every iteration to decrease the Loss Function. If the intermediate loss function values need to be printed, set printIntermediateLossFunctionValues to true
def runNewIteration(previousLossFunctionvalue, currentAlgorithmIterationValues, K, threshold, printIntermediateLossFunctionValues = False):
    if printIntermediateLossFunctionValues == True:
        print(previousLossFunctionvalue)
    # Update the centroids by using the improveLossFunction() function
    currentAlgorithmIterationValues.improveLossFunctionValue()
    # Determine the value of the loss function after the new centroid update
    newLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()
    # Since newLossFunctionValue <= previousLossFuncitonvalue we get a decreasing number, we stop when they're very close i.e. their fraction is very small
    if (previousLossFunctionvalue/newLossFunctionValue < threshold):
        print(f"Final loss function value for K = {K}  is {newLossFunctionValue}")
        # Return None when the ratio is below the Treshold
        return None
    # update the loss function value to be able to compare the new value to the old value
    return newLossFunctionValue

def improveUntilTresholdReached(K, treshold, printIntermediateLossFunctionValues=False):
    # Update to first Iteration (this differs from other iteration since it has to construct start centroids)
    currentAlgorithmIterationValues = runFirstIteration(K)
    # Calculate the start loss function value after the first iteration
    startLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()
    # set previousLossFuncitonvalue to startLossFunctionValue so they can be compared in the for loop
    previousLossFunctionvalue = startLossFunctionValue
    # loop from 0 to a very higher number so the centroids can be updated in each loop until the stopping criterium is reached
    while (previousLossFunctionvalue != None):
        # update each previous loss function value with a new improved one
        previousLossFunctionvalue = runNewIteration(previousLossFunctionvalue, currentAlgorithmIterationValues, K, treshold, printIntermediateLossFunctionValues)

# improveUntilTresholdReached(10, 1.000_01, True)