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
    currentRunNumber = 1
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
    currentRunNumber = 1
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
    if (currentRunNumber == 41):
        return None
    return newLossFunctionValue

K = 10 

AmountOfIterationsList = []
for i in range(0, 41):
    AmountOfIterationsList.append(i)

for i in range(10):
    improveRandomCentroidStartForFourtySteps(K)

print(randomCentroidsList)

for dataSetOf40Points in randomCentroidsList:
    plt.plot(AmountOfIterationsList, dataSetOf40Points, color='blue', linestyle='dashed', linewidth = 1,
        marker='o', markerfacecolor='blue', markersize=2, label="Start centroids determined at random")


for i in range(10):
    improveKPlusplusCentroidStartForFourtySteps(K)


for dataSetOf40Points in KPlusPlusList:
    plt.plot(AmountOfIterationsList, dataSetOf40Points, color='red', linestyle='dashed', linewidth = 1,
        marker='o', markerfacecolor='red', markersize=2, label="Start centroids determined by using K++")

 
    

plt.legend(loc="upper right")

# naming the x axis
plt.xlabel('Value of the loss function')
# naming the y axis
plt.ylabel('Number of iterations')

# giving a title to my graph
plt.title('Value of the loss function after each given number of iterations')

# # function to show the plot
plt.show()