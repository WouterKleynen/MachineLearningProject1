import pandas as pd
import numpy as np
from KMeansClustering import KMeansClustering

class KernelKMeansClustering(KMeansClustering):
    
    # Setup for the start parameters
    def __init__(self, dataFilePath, K, kernel):
        
        super(KernelKMeansClustering, self).__init__(dataFilePath, K)
        self.kernel = kernel
        self.kAccentMatrix = np.zeros((self.amountOfRows, self.amountOfClusters)) # Row i consists of the distances of the kAccent values of point i.
        # The inherited self.dataWithoutIDMatrix is set to standardizedDataWithoutID in the first step of the firstiteration: self.standardizeData()
        self.IDClusterDictionary = {}
        
    def standardize(self, columnIndex):                                                                
        mu = np.average(self.data[:, columnIndex])
        sigma = np.std(self.data[:, columnIndex])
        Z = (self.data[:, columnIndex] - mu)/sigma
        return Z
    
    def standardizeData(self):                                                                          
        standardizedMatrix = np.zeros((self.amountOfRows, self.amountOfColumns))
        for columnIndex in range(0, self.amountOfColumns):
            standardizedMatrix[:, columnIndex] = self.standardize(columnIndex)
        self.dataWithoutIDMatrix = standardizedMatrix[:, 1:]                                 
        pd.DataFrame(self.dataWithoutIDMatrix).to_csv("Dataset\standardizedData.csv",index=False, header=False)

    def getIndexMinimumCluster(self, rowIndex):                                    # Given row index i, it gets the index of the minimum of row i of kAccentMatrix
        return np.argmin(self.kAccentMatrix[rowIndex])
    
    def setKAccentValues(self):
            clusterVectorSizes = self.getClusterVectorSizesVector()
            K3Vector           = self.sumOfKernelOfAllPointsInClusterVector()                     # get the third K expression in the formula. This is the same for every cluster point and very expensive in computing so calculate beforehand
            for rowIndex in range (0, self.amountOfRows):
                for clusterIndex in range(self.amountOfClusters):
                    point        = self.dataWithoutIDMatrix[rowIndex]                       # get normalized point
                    K3Value      = K3Vector[clusterIndex]                                   
                    clusterSize  = clusterVectorSizes[clusterIndex]
                    self.kAccentMatrix[rowIndex, clusterIndex] = self.getKAccentValue(point, clusterIndex, K3Value, clusterSize)
    
    def setClusterDictionary(self):                                                # For each key (clusterIndex) in the clusterDictionary, determines which points are closests to the centroid of that cluster, then it adds the ID's of these points to the clusterVector being the value belonging to the clusterIndex key.
        self.emptyClusterDictionary()                                              # empty the old cluster vectors.
        for rowIndex in range(self.amountOfRows):                                  # iterate over all the points.
            id = self.idVector[rowIndex]                                           # Get the ID belonging to each point.
            closestClusterIndex = self.getIndexClosestCentroid(rowIndex)           # Get the index of closest centroid by finding the minimum of row i of centroidToPointsDistancesMatrix.
            self.clusterDictionary[closestClusterIndex].append(id)
            self.IDClusterDictionary[id] = closestClusterIndex
    
    def setKernelClusterDictionary(self):                                          
        self.emptyClusterDictionary()                                               # empty the old cluster vectors.
        for rowIndex in range(self.amountOfRows):                                   # iterate over all the points.
            id = self.idVector[rowIndex]                                            # Get the ID belonging to each point.
            closestClusterIndex = self.getIndexMinimumCluster(rowIndex)             # Get the index of the minimum entry of row i that is the, that index is the cluster index of the cluster that point i is put in.
            self.clusterDictionary[closestClusterIndex].append(id)
            self.IDClusterDictionary[id] = closestClusterIndex

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
        if clusterSize == 0:   
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
            pointInCluster = self.getPointFromID(ID)                                # gets point from nonStandardizedIDVector
            totalSum      += self.kernel(point, pointInCluster)
        return totalSum
    
    def fillClusterCSV(self):                                                         # Fill each Clusters CSV file with the points belonging to it. 
        for clusterIndexKey in self.clusterDictionary:
            clusterIDVector = self.clusterDictionary[clusterIndexKey]
            for pointID in clusterIDVector:
                dataPoint = self.getPointFromIDWithID(pointID)
                dataFrame = pd.DataFrame(dataPoint)
                dataFrame.T.to_csv(f'Dataset\KernelClusteredData\Cluster{clusterIndexKey}.csv', mode='a', index=False, header=False)
    
    def firstIteration(self):
        self.standardizeData()
        self.kMeansPlusPlusMethod()
        self.setDistanceOfPointsToCentroidsMatrix()
        self.setClusterDictionary()
        
    def improveKernelLossFunctionValue(self):
        self.setKAccentValues()
        self.setKernelClusterDictionary()
        self.setCentroids()