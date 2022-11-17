import pandas as pd
import numpy as np
import scipy
from KMeansClustering import KMeansClustering

class NEW(KMeansClustering):
    
    # Setup for the start parameters
    def __init__(self, dataFilePath, K, kernel):
        
        super(NEW, self).__init__(dataFilePath, K)
        # The inherited self.dataWithoutIDMatrix is set to standardizedDataWithoutID in the first step of the firstiteration: self.standardizeData()
        
        self.kernel                          = kernel                       # Kernel function that is being used
        self.nonStandardizedData             = self.data                    # Nonstandardized data with ID
        self.nonStandardizedDataWithoutID    = self.dataWithoutIDMatrix     # Nonstandardized data without ID
        self.nonStandardizedIDVector         = self.data[:, 0]                                                # Only ID's are extracted.
        self.standardizedData                = np.array((self.amountOfRows, self.amountOfColumns))  # Standardized data
        self.standardizedDataWithoutID       = np.array((self.amountOfRows, self.amountOfColumns - 1))  # Standardized data without ID
        self.kAccentMatrix                   = np.zeros((self.amountOfRows, self.amountOfClusters)) # Row i consists of the distances of the kAccent values of point i.
    
    def getPointIndexFromId(self, id):                                              
        return np.where(self.nonStandardizedIDVector == id)[0][0]                   # You want the ID's to remain the same so nonStandardized
    
    def getPointFromPointIndex(self, pointIndex):                                   # Gets the row of standardizedDataWithoutID belonging to the given point index value. 
        return self.standardizedDataWithoutID[pointIndex, :]
    
    def getPointFromID(self, id):                                                   # Gets the row of standardizedDataWithoutID belonging to the given ID value. 
        return self.getPointFromPointIndex(self.getPointIndexFromId(id))
    
    def standardize(self, columnIndex):                                                                
        mu = np.average(self.data[:, columnIndex])
        sigma = np.std(self.data[:, columnIndex])
        Z = (self.data[:, columnIndex] - mu)/sigma
        return Z
    
    def standardizeData(self):                                                                          
        standardizedMatrix = np.zeros((self.nonStandardizedData.shape[0], self.nonStandardizedData.shape[1]))
        numberOfColumns    = self.nonStandardizedData.shape[1]
        for columnIndex in range(0, numberOfColumns):
            standardizedMatrix[:, columnIndex] = self.standardize(columnIndex)
        self.standardizedData          = standardizedMatrix
        self.standardizedDataWithoutID = standardizedMatrix[:, 1:]                                       
        pd.DataFrame(self.standardizedDataWithoutID).to_csv("Dataset\standardizedData.csv",index=False, header=False)

    def kMeansPlusPlusMethod(self):                                                 
        C = [self.nonStandardizedDataWithoutID[0]]                                         # Use the nonStandardized data to get the start centroids
        for _ in range(1, self.amountOfClusters):
            D2 = scipy.array([min([scipy.inner(c-x,c-x) for c in C]) for x in self.nonStandardizedDataWithoutID])
            probs = D2/D2.sum()
            cumprobs = probs.cumsum()
            r = scipy.rand()
            for j,p in enumerate(cumprobs):
                if r < p:
                    i = j
                    break
            C.append(self.nonStandardizedDataWithoutID[i])                                    # sets row i as centroid i
        self.centroidsMatrix = np.array(C)

    def setDistanceOfPointsToCentroidsMatrix(self):                                           # Sets the centroidToPointsDistancesMatrix (N x K) entries, where row i stores the distance of point i to each cluster. Or similarly where column j stores the distance of all points to cluster j.
        for rowIndex in range (0, self.amountOfRows):
            for centroidIndex in range(self.amountOfClusters):
                nonStandardizedPoint    = self.nonStandardizedDataWithoutID[rowIndex]         # Since the centroids are non standardized, the points have to be non standardized as well
                nonStandardizedCentroid = self.centroidsMatrix[centroidIndex]
                self.centroidToPointsDistancesMatrix[rowIndex, centroidIndex] = self.getEuclideanDistance(nonStandardizedPoint, nonStandardizedCentroid)

    def setClusterDictionaryFirstRun(self):                                                   # For each key (clusterIndex) in the clusterDictionary, determines which points are closests to the centroid of that cluster, then it adds the ID's of these points to the clusterVector being the value belonging to the clusterIndex key.
        self.emptyClusterDictionary()                                                         # empty the old cluster vectors.
        for rowIndex in range(self.amountOfRows):                                             # iterate over all the points.
            id = self.idVector[rowIndex]                                                      # Get the ID belonging to each point.
            closestClusterIndex = self.getIndexClosestCentroid(rowIndex)                      # Get the index of closest centroid by finding the minimum of row i of centroidToPointsDistancesMatrix.
            self.clusterDictionary[closestClusterIndex].append(id)

    def setKAccentValues(self):
        clusterVectorSizes = self.getClusterVectorSizesVector()
        K3Vector           = self.sumOfKernelOfAllPointsInClusterVector()
        for rowIndex in range (0, self.amountOfRows):
            for clusterIndex in range(self.amountOfClusters):
                point        = self.dataWithoutIDMatrix[rowIndex]
                K3Value      = K3Vector[clusterIndex]
                clusterSize  = clusterVectorSizes[clusterIndex]
                self.kAccentMatrix[rowIndex, clusterIndex] = self.getKAccentValue(point, clusterIndex, K3Value, clusterSize)
    
    def sumOfKernelOfAllPointsInClusterVector(self):
        sumOfKernelOfAllPointsInClusterVector = []
        for clusterIndex in range(0, self.amountOfClusters):
            value = self.sumOfKernelOfAllPointsInCluster(clusterIndex)
            sumOfKernelOfAllPointsInClusterVector.append(value)
        return sumOfKernelOfAllPointsInClusterVector
    
    def sumOfKernelOfAllPointsInCluster(self, clusterIndex):
        totalSum = 0
        clusterVector = self.getClusterVector(clusterIndex)
        for ID in clusterVector:
            pointInCluster = self.getPointFromID(ID)
            for IDAgain in clusterVector:
                pointInclusterAgain = self.getPointFromID(IDAgain)
                totalSum += self.kernel(pointInCluster, pointInclusterAgain)
        return totalSum
    
    def getKAccentValue(self, point, clusterIndex, sumOfKernelOfAllPointsInCluster, clusterSize):
        if clusterSize == 0:   # If cluster size is 0 we have an issue. Think about how to fix this!
            print("TEST!")
            return None
        firstTerm                          = self.kernel(point, point)
        sumOfGaussianDistanceWithPoint     = self.sumOfKernelOfPoint(point, clusterIndex)
        secondTerm = (- 2.0 / clusterSize) * sumOfGaussianDistanceWithPoint
        thirdTerm = (1.0 / clusterSize**2) * sumOfKernelOfAllPointsInCluster
        value = firstTerm + secondTerm + thirdTerm
        return value
    
    def sumOfKernelOfPoint(self, point, clusterIndex):
        totalSum = 0
        clusterVector = self.getClusterVector(clusterIndex)
        for ID in clusterVector:
            pointInCluster = self.getPointFromID(ID)                            # gets point from nonStandardizedIDVector
            totalSum      += self.kernel(point, pointInCluster)
        return totalSum
    


    
    
    
    def firstIteration(self):
        self.kMeansPlusPlusMethod()
        self.setDistanceOfPointsToCentroidsMatrix()
        self.setClusterDictionaryFirstRun()