from AlgorithmClass import *
import matplotlib.pyplot as plt

# This file contains functions to asses which start centroid method works best for our data: K++ or random. 
# In order to asses this we iterate from K=2 to K=41 and plot 10 full iterations of the K Means Euclidean 
# Algorithm using both start methods where each iteration consists of 40 lossfunction improvements until 
# it terminates. This way we asses which start method has the lowest average loss function at the end. 

dataSetFilePath = 'Dataset/InputData.csv'

randomCentroidsList = [] 

# First iteration which uses random start centroids
def startFirstIterationRandom(K):
    currentAlgorithmIterationValues = KMeansClusteringEuclidean(dataSetFilePath, K)
    currentAlgorithmIterationValues.runFirstIterationRandom()
    return currentAlgorithmIterationValues

# Minor changes w.r.t improveUntilTresholdReached() consisting of appending to list for plotting and removing redundant variables 
def improveRandomCentroidStartForFourtySteps(K):
    randomCentroidsListForK = []
    currentRunNumber = 0
    currentAlgorithmIterationValues = startFirstIterationRandom(K)
    startLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()
    lossFunctionvalue = startLossFunctionValue
    while (lossFunctionvalue != None):
        lossFunctionvalue = runNewIterationWithFixedEnd(currentAlgorithmIterationValues, currentRunNumber)
        randomCentroidsListForK.append(lossFunctionvalue)
        currentRunNumber += 1
    randomCentroidsList.append(randomCentroidsListForK)
    return lossFunctionvalue


KPlusPlusList = []

# First iteration which uses K++ start centroids
def startFirstIterationKplusplus(K):
    currentAlgorithmIterationValues = KMeansClusteringEuclidean(dataSetFilePath, K)
    currentAlgorithmIterationValues.runFirstIterationKPlusPlus()
    return currentAlgorithmIterationValues


# Minor changes w.r.t improveUntilTresholdReached() consisting of appending to list for plotting and removing redundant variables 
def improveKPlusplusCentroidStartForFourtySteps(K):
    print("\n")
    KPlusPlusCentroidsListForK = []
    currentRunNumber = 0
    currentAlgorithmIterationValues = startFirstIterationKplusplus(K)
    startLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()
    lossFunctionvalue = startLossFunctionValue
    while (lossFunctionvalue != None):
        lossFunctionvalue = runNewIterationWithFixedEnd(currentAlgorithmIterationValues, currentRunNumber)
        KPlusPlusCentroidsListForK.append(lossFunctionvalue)
        currentRunNumber += 1
    KPlusPlusList.append(KPlusPlusCentroidsListForK)
    return lossFunctionvalue

# Changes w.r.t normal runNewIteration is removing the treshold and use a hardcoded amount of iterations instead and removing redundant variables
def runNewIterationWithFixedEnd(currentAlgorithmIterationValues, currentRunNumber):
    currentAlgorithmIterationValues.improveLossFunctionValue()
    newLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()
    if (currentRunNumber == 39):
        finalLoss.append(newLossFunctionValue)
        return None
    return newLossFunctionValue

finalLoss = []

# Iterate from K=2 to K=41
for K in range(2, 41):
    plt.clf()
    
    
    randomCentroidsList = [] # Clear the plot list values after each plot
    KPlusPlusList = []

    finalLoss.append("RANDOM")
    finalLoss.append(f"K = {K}")
    
    AmountOfIterationsList = []
    
    for i in range(0, 40):
        AmountOfIterationsList.append(i)

    for i in range(10):
        improveRandomCentroidStartForFourtySteps(K) # run 10 iterations of the K means Euclidean algorithm with random start centroids

    for index in range(0, len(randomCentroidsList) - 1): # Create plots
        plt.plot(AmountOfIterationsList, randomCentroidsList[index], color='blue', linestyle='dashed', linewidth = 1,
            marker='o', markerfacecolor='blue', markersize=2)

    # Last plot seperate, because this one includes a label. If it was in the loop the label would be added len(randomCentroidsList) - 1 times
    plt.plot(AmountOfIterationsList, randomCentroidsList[len(randomCentroidsList) - 1], color='blue', linestyle='dashed', linewidth = 1,
            marker='o', markerfacecolor='blue', markersize=2, label="Start centroids are determined at random")

    finalLoss.append("SWITCHED TO K++")

    for i in range(10):
        improveKPlusplusCentroidStartForFourtySteps(K) # run 10 iterations of the K means Euclidean algorithm with K++ start centroids


    for index in range(0, len(KPlusPlusList) - 1): # Create Plots
        plt.plot(AmountOfIterationsList, KPlusPlusList[index], color='red', linestyle='dashed', linewidth = 1,
            marker='o', markerfacecolor='red', markersize=2)

    # Last plot seperate, because this one includes a label. If it was in the loop the label would be added len(randomCentroidsList) - 1 times
    plt.plot(AmountOfIterationsList, KPlusPlusList[len(KPlusPlusList) - 1], color='red', linestyle='dashed', linewidth = 1,
            marker='o', markerfacecolor='red', markersize=2, label="Start centroids are determined by using K++")
        

    plt.legend(loc="upper right")

    plt.xlabel('Number of iterations')
    plt.ylabel('Value of the loss function')
    plt.title(f'Loss function value after each given number of iterations where K = {K}')
    plt.savefig(f'k={K}.png')
    plt.close()
    
print(finalLoss)