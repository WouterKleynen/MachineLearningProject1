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
    if (currentRunNumber == 40):
        return None
    return newLossFunctionValue

K = 10 

AmountOfIterationsList = []
for i in range(0, 40):
    AmountOfIterationsList.append(i)

for i in range(10):
    improveRandomCentroidStartForFourtySteps(K)

print(randomCentroidsList)

for dataSetOf40Points in randomCentroidsList:
    plt.plot(AmountOfIterationsList, dataSetOf40Points, color='black', linestyle='dashed', linewidth = 1,
        marker='o', markerfacecolor='red', markersize=2)



# for dataSetOf40Points in randomCentroidsList:
#     plt.plot(AmountOfIterationsList, dataSetOf40Points, color='black', linestyle='dashed', linewidth = 1,
#         marker='o', markerfacecolor='red', markersize=2)


# for i in range(10):
#     improveKPlusplusCentroidStartForFourtySteps(K)


# for dataSetOf40Points in KPlusPlusList:
#     plt.plot(AmountOfIterationsList, dataSetOf40Points, color='black', linestyle='dashed', linewidth = 3,
#         marker='o', markerfacecolor='blue', markersize=12)
    
    
# # function to show the plot
plt.show()


# plt.plot(xAxisValyes, lossFunctionValuesForK, color='black', linestyle='dashed', linewidth = 3,
#         marker='o', markerfacecolor='red', markersize=12)

# # naming the x axis
# plt.xlabel('Number of clusters (K)')
# # naming the y axis
# plt.ylabel('Loss Function value')

# # giving a title to my graph
# plt.title('Loss Function values for each Number of Clusters')

# # function to show the plot
# plt.show()