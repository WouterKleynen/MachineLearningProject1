import matplotlib.pyplot as plt
from KMeansClusteringEuclidean import *

lossFunctionValuesForK =  []

dataSetFilePath = 'Dataset/InputData.csv'                                                           # Set data File path to that of the assignment data sheet.

# Is called to run the first iteration. The first iteration differs from other iteration since it has to construct start centroids.
def runFirstIteration(K):
    currentAlgorithmIterationValues = KMeansClusteringEuclidean(dataSetFilePath, K)                 # Create an instance of the KMeansClusteringEuclidean class.
    currentAlgorithmIterationValues.firstIteration()                                                # Set the start Centroids and fill each cluster with its closest data points for the first run of the algorithm.
    return currentAlgorithmIterationValues

def improveUntilTresholdReachedForOptimalK(K, printIntermediateLossFunctionValues=False):
    # Update to first Iteration (this differs from other iteration since it has to construct start centroids)
    currentAlgorithmIterationValues = runFirstIteration(K)
    # Calculate the start loss function value after the first iteration
    startLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()
    # set previousLossFuncitonvalue to startLossFunctionValue so they can be compared in the for loop
    previousLossFunctionvalue = startLossFunctionValue
    # loop from 0 untill the iteration that the treshold is reached: when previousLossFunctionvalue == None
    while (previousLossFunctionvalue != None):
        # update each previous loss function value with a new improved one
        previousLossFunctionvalue = runNewIterationForOptimalK(previousLossFunctionvalue, currentAlgorithmIterationValues, K)


# Is called in every iteration to decrease the Loss Function. If the intermediate loss function values need to be printed, set printIntermediateLossFunctionValues to true
def runNewIterationForOptimalK(previousLossFunctionvalue, currentAlgorithmIterationValues, K):
    # Update the centroids by using the improveLossFunction() function
    currentAlgorithmIterationValues.improveLossFunctionValue()
    # Determine the value of the loss function after the new centroid update
    newLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()
    # Since newLossFunctionValue <= previousLossFuncitonvalue we get a decreasing number, we stop when they're very close i.e. their fraction is very small
    if (previousLossFunctionvalue == newLossFunctionValue):
        print(f"Final loss function value for K = {K} is {newLossFunctionValue}")
        lossFunctionValuesForK.append(newLossFunctionValue)
        # Return None when the ratio is below the Treshold
        return None
    # update the loss function value to be able to compare the new value to the old value
    return newLossFunctionValue


xAxisValyes = []
for K in range(1, 40):
    xAxisValyes.append(K)
    improveUntilTresholdReachedForOptimalK(K, 1.000_01)

# An idea to look at relative improvements and for which K we still get meaning full data

print(xAxisValyes)
print(lossFunctionValuesForK)
# plotting the points 

plt.plot(x, y, color='black', linestyle='dashed', linewidth = 2,
         marker='o', markerfacecolor='red', markersize=6)
  
# naming the x axis
plt.xlabel('Number of clusters (K)')
# naming the y axis
plt.ylabel('Final loss function value')
  
# giving a title to my graph
plt.title('Final loss function value for each number of clusters')
  
# function to show the plot
plt.show()