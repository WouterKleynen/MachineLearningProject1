from AlgorithmClass import *
import matplotlib.pyplot as plt

dataSetFilePath = 'Dataset/InputData.csv'

randomCentroidsList = []

def startFirstIterationRandom(K):
    currentAlgorithmIterationValues = KMeansClusteringEuclidean(dataSetFilePath, K)
    currentAlgorithmIterationValues.runFirstIterationRandom()
    return currentAlgorithmIterationValues

def improveRandomCentroidStartForFourtySteps(K):
    print("\n")
    randomCentroidsListForK = []
    currentRunNumber = 0
    currentAlgorithmIterationValues = startFirstIterationRandom(K)
    startLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()
    lossFunctionvalue = startLossFunctionValue
    while (lossFunctionvalue != None):
        lossFunctionvalue = runNewIterationWithFixedEnd(lossFunctionvalue, currentAlgorithmIterationValues, currentRunNumber)
        randomCentroidsListForK.append(lossFunctionvalue)
        currentRunNumber += 1
    randomCentroidsList.append(randomCentroidsListForK)
    return lossFunctionvalue


KPlusPlusList = []

def startFirstIterationKplusplus(K):
    currentAlgorithmIterationValues = KMeansClusteringEuclidean(dataSetFilePath, K)
    currentAlgorithmIterationValues.runFirstIterationKPlusPlus()
    return currentAlgorithmIterationValues

    
def improveKPlusplusCentroidStartForFourtySteps(K):
    print("\n")
    KPlusPlusCentroidsListForK = []
    currentRunNumber = 0
    currentAlgorithmIterationValues = startFirstIterationKplusplus(K)
    startLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()
    lossFunctionvalue = startLossFunctionValue
    while (lossFunctionvalue != None):
        lossFunctionvalue = runNewIterationWithFixedEnd(lossFunctionvalue, currentAlgorithmIterationValues, currentRunNumber)
        KPlusPlusCentroidsListForK.append(lossFunctionvalue)
        currentRunNumber += 1
    KPlusPlusList.append(KPlusPlusCentroidsListForK)
    return lossFunctionvalue

def runNewIterationWithFixedEnd(previousLossFunctionvalue, currentAlgorithmIterationValues, currentRunNumber):
    print(f"current loss fuction value = {previousLossFunctionvalue}")
    currentAlgorithmIterationValues.improveLossFunctionValue()
    newLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()
    if (currentRunNumber == 39):
        return None
    return newLossFunctionValue

for K in range(1, 31):

    AmountOfIterationsList = []
    for i in range(0, 40):
        AmountOfIterationsList.append(i)

    for i in range(10):
        improveRandomCentroidStartForFourtySteps(K)

    for index in range(0, len(randomCentroidsList) - 1):
        plt.plot(AmountOfIterationsList, randomCentroidsList[index], color='blue', linestyle='dashed', linewidth = 1,
            marker='o', markerfacecolor='blue', markersize=2)

    plt.plot(AmountOfIterationsList, randomCentroidsList[len(randomCentroidsList) - 1], color='blue', linestyle='dashed', linewidth = 1,
            marker='o', markerfacecolor='blue', markersize=2, label="Start centroids are determined at random")


    for i in range(10):
        improveKPlusplusCentroidStartForFourtySteps(K)


    for index in range(0, len(KPlusPlusList) - 1):
        plt.plot(AmountOfIterationsList, KPlusPlusList[index], color='red', linestyle='dashed', linewidth = 1,
            marker='o', markerfacecolor='red', markersize=2)

    plt.plot(AmountOfIterationsList, KPlusPlusList[len(KPlusPlusList) - 1], color='red', linestyle='dashed', linewidth = 1,
            marker='o', markerfacecolor='red', markersize=2, label="Start centroids are determined by using K++")
        

    plt.legend(loc="upper right")

    # naming the x axis
    plt.xlabel('Number of iterations')
    # naming the y axis
    plt.ylabel('Value of the loss function')


    # giving a title to my graph
    plt.title('Value of the loss function after each given number of iterations given K = {K}')

    plt.savefig(f'k={K}.png')

    # # # function to show the plot
    # plt.show()