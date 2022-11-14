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
    if ( currentRunNumber > 40):
        return None
    return newLossFunctionValue

KList = []
for i in range(0, 40):
    KList.append(i)

K = 10 
for i in range(10):
    improveKPlusplusCentroidStartForFourtySteps(K)

for i in range(10):
    improveRandomCentroidStartForFourtySteps(K)

for dataSetOf40Points in randomCentroidsList:
    plt.plot(KList, dataSetOf40Points, color='black', linestyle='dashed', linewidth = 3,
        marker='o', markerfacecolor='red', markersize=12)


for dataSetOf40Points in KPlusPlusList:
    plt.plot(KList, dataSetOf40Points, color='black', linestyle='dashed', linewidth = 3,
        marker='o', markerfacecolor='blue', markersize=12)
    
    
# # function to show the plot
plt.show()

print(randomCentroidsList)
print(KPlusPlusList)


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