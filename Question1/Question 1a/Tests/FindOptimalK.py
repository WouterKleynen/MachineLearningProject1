import matplotlib.pyplot as plt
from RunAlgorithm import runFirstIteration

# lossFunctionValuePerKList = [260193326.7624317, 198061130.83388263, 155563638.1885101, 125341947.04330947, 121258416.71073835, 101505029.84715219, 100801930.51575494, 89733070.97581671, 89195408.9602358]
# xAxisValyes               = [1, 2, 3, 4, 5, 6, 7, 8, 9]

lossFunctionValuesForK =  []

def improveUntilTresholdReachedForOptimalK(K, treshold, printIntermediateLossFunctionValues=False):
    # Update to first Iteration (this differs from other iteration since it has to construct start centroids)
    currentAlgorithmIterationValues = runFirstIteration(K)
    # Calculate the start loss function value after the first iteration
    startLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()
    # set previousLossFuncitonvalue to startLossFunctionValue so they can be compared in the for loop
    previousLossFunctionvalue = startLossFunctionValue
    # loop from 0 untill the iteration that the treshold is reached: when previousLossFunctionvalue == None
    while (previousLossFunctionvalue != None):
        # update each previous loss function value with a new improved one
        previousLossFunctionvalue = runNewIterationForOptimalK(previousLossFunctionvalue, currentAlgorithmIterationValues, K, treshold, printIntermediateLossFunctionValues)


# Is called in every iteration to decrease the Loss Function. If the intermediate loss function values need to be printed, set printIntermediateLossFunctionValues to true
def runNewIterationForOptimalK(previousLossFunctionvalue, currentAlgorithmIterationValues, K, threshold, printIntermediateLossFunctionValues = False):
    if printIntermediateLossFunctionValues == True:
        print(f"current loss fuction value = {previousLossFunctionvalue}")
    # Update the centroids by using the improveLossFunction() function
    currentAlgorithmIterationValues.improveLossFunctionValue()
    # Determine the value of the loss function after the new centroid update
    newLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()
    # Since newLossFunctionValue <= previousLossFuncitonvalue we get a decreasing number, we stop when they're very close i.e. their fraction is very small
    if (previousLossFunctionvalue/newLossFunctionValue < threshold):
        print(f"Final loss function value for K = {K} is {newLossFunctionValue}")
        lossFunctionValuesForK.append(newLossFunctionValue)
        # Return None when the ratio is below the Treshold
        return None
    # update the loss function value to be able to compare the new value to the old value
    return newLossFunctionValue


xAxisValyes = []
for K in range(1, 10):
    xAxisValyes.append(K)
    improveUntilTresholdReachedForOptimalK(K, 1.000_01)

# An idea to look at relative improvements and for which K we still get meaning full data

print(xAxisValyes)
print(lossFunctionValuesForK)
# plotting the points 

plt.plot(xAxisValyes, lossFunctionValuesForK, color='black', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='red', markersize=12)
  
# naming the x axis
plt.xlabel('Number of clusters (K)')
# naming the y axis
plt.ylabel('Loss Function value')
  
# giving a title to my graph
plt.title('Loss Function values for each Number of Clusters')
  
# function to show the plot
plt.show()