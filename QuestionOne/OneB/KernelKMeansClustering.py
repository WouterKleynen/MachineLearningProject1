import pandas as pd
import numpy as np
from Tools import getGaussianDistance
from KMeansClustering import KMeansClustering


class KernelKMeansClustering(KMeansClustering):
    
    
    # Setup for the start parameters
    def __init__(self, dataFilePath, K, kernel):
        
        super(KernelKMeansClustering, self).__init__(dataFilePath, K)
        self.fullOriginalData                = self.data                    # Nonstandardized data with ID
        self.kernel                          = kernel
        self.originalDataWithoudID           = self.dataWithoutIDMatrix     # Nonstandardized data without ID
        self.standardizedData                = self.dataWithoutIDMatrix     # Nonstandardized data without ID
        self.kDistanceMatrix                 = np.zeros((self.amountOfRows, self.amountOfClusters))           # Row i consists of the distances of point i to each cluster.
    
    #########################################################################################################
    # Standardizing functions
    #########################################################################################################
    
    def standardize(self, columnIndex):                                                                
        mu = np.average(self.data[:, columnIndex])
        sigma = np.std(self.data[:, columnIndex])
        Z = (self.data[:, columnIndex] - mu)/sigma
        return Z

    def standardizeData(self):                                                                          
        standardizedMatrix = np.zeros((self.data.shape[0], self.data.shape[1]))
        numberOfColumns = self.data.shape[1]
        for columnIndex in range(0, numberOfColumns):
            standardizedMatrix[:, columnIndex] = self.standardize(columnIndex)
        standardizedMatrix = standardizedMatrix[:, 1:]                                       
        pd.DataFrame(standardizedMatrix).to_csv("Dataset\standardizedData.csv",index=False, header=False)
        self.standardizedData    = standardizedMatrix
        self.dataWithoutIDMatrix = standardizedMatrix
    
    #########################################################################################################
    # Getter functions
    #########################################################################################################
    
    def getIndexMinimumCluster(self, rowIndex):                                    # Given row index i, it gets the index of the minimum of row i of kDistanceMatrix
        return np.argmin(self.kDistanceMatrix[rowIndex])
    
    #########################################################################################################
    # Setter functions
    #########################################################################################################
    
    def setClusterDictionaryFirstRun(self):                                        # For each key (clusterIndex) in the clusterDictionary, determines which points are closests to the centroid of that cluster, then it adds the ID's of these points to the clusterVector being the value belonging to the clusterIndex key.
        self.emptyClusterDictionary()                                              # empty the old cluster vectors.
        for rowIndex in range(self.amountOfRows):                                  # iterate over all the points.
            id = self.idVector[rowIndex]                                           # Get the ID belonging to each point.
            closestClusterIndex = self.getIndexClosestCentroid(rowIndex)           # Get the index of closest centroid by finding the minimum of row i of centroidToPointsDistancesMatrix.
            self.clusterDictionary[closestClusterIndex].append(id)

    def setClusterDictionary(self):                                                # For each key (clusterIndex) in the clusterDictionary, determines which points are closests to the centroid of that cluster, then it adds the ID's of these points to the clusterVector being the value belonging to the clusterIndex key.
        self.emptyClusterDictionary()                                              # empty the old cluster vectors.
        for rowIndex in range(self.amountOfRows):                                  # iterate over all the points.
            id = self.idVector[rowIndex]                                           # Get the ID belonging to each point.
            closestClusterIndex = self.getIndexMinimumCluster(rowIndex)            # Get the index of closest centroid by finding the minimum of row i of centroidToPointsDistancesMatrix.
            self.clusterDictionary[closestClusterIndex].append(id)
    
    def setDistanceOfPointsToCentroidsMatrix(self):                                 # Sets the centroidToPointsDistancesMatrix (N x K) entries, where row i stores the distance of point i to each cluster. Or similarly where column j stores the distance of all points to cluster j.
        for rowIndex in range (0, self.amountOfRows):
            for centroidIndex in range(self.amountOfClusters):
                self.centroidToPointsDistancesMatrix[rowIndex, centroidIndex] = self.getEuclideanDistance(self.dataWithoutIDMatrix[rowIndex], self.centroidsMatrix[centroidIndex])
    
    def setKAccentValues(self):
        clusterVectorSizes = self.getClusterVectorSizesVector()
        K3Vector           = self.sumOfKernelOfAllPointsInClusterVector()
        for rowIndex in range (0, self.amountOfRows):
            for clusterIndex in range(self.amountOfClusters):
                point = self.dataWithoutIDMatrix[rowIndex]
                K3Value = K3Vector[clusterIndex]
                clusterSize = clusterVectorSizes[clusterIndex]
                self.kDistanceMatrix[rowIndex, clusterIndex] = self.getKAccentValueNew(point, clusterIndex, K3Value, clusterSize)

    #########################################################################################################
    # Calculation functions
    #########################################################################################################

    def sumOfKernelOfPoint(self, point, clusterIndex):
        totalSum = 0
        clusterVector = self.getClusterVector(clusterIndex)
        for ID in clusterVector:
            pointInCluster = self.getPointFromID(ID)
            totalSum      += self.kernel(point, pointInCluster)
        return totalSum
    
    def sumOfKernelOfAllPointsInCluster(self, clusterIndex):
        totalSum = 0
        clusterVector = self.getClusterVector(clusterIndex)
        for ID in clusterVector:
            pointInCluster = self.getPointFromID(ID)
            for IDAgain in clusterVector:
                pointInclusterAgain = self.getPointFromID(IDAgain)
                totalSum            += self.kernel(pointInCluster, pointInclusterAgain)
        return totalSum

    def sumOfKernelOfAllPointsInClusterVector(self):
        sumOfKernelOfAllPointsInClusterVector = []
        for clusterIndex in range(0, self.amountOfClusters):
            value = self.sumOfKernelOfAllPointsInCluster(clusterIndex)
            sumOfKernelOfAllPointsInClusterVector.append(value)
        return sumOfKernelOfAllPointsInClusterVector
    
    def getKAccentValueNew(self, point, clusterIndex, sumOfKernelOfAllPointsInCluster, clusterSize):
        if clusterSize == 0:
            return None
        firstTerm                          = self.kernel(point, point)
        sumOfGaussianDistanceWithPoint     = self.sumOfKernelOfPoint(point, clusterIndex)
        secondTerm = (- 2.0 / clusterSize) * sumOfGaussianDistanceWithPoint
        thirdTerm = (1.0 / clusterSize**2) * sumOfKernelOfAllPointsInCluster
        value = firstTerm + secondTerm + thirdTerm
        return value
    
    def calculateLossFunctionValue(self):                                           # Calculate the sum of all the distances of the data points to the centers of the clusters they belong to.        
        self.dataWithoutIDMatrix = self.originalDataWithoudID
        loss = 0
        for clusterIndex in range(0, self.amountOfClusters):
            clusterVector = self.getClusterVector(clusterIndex)
            centroidVector = self.getCentroidVector(clusterIndex)
            for id in clusterVector:
                point = self.getPointFromID(id)
                loss += self.getEuclideanDistance(centroidVector, point)
        self.dataWithoutIDMatrix = self.standardizedData
        return loss
    
    def setCentroids(self):                                                         # Sets the Centroids of all clusters by calculatin the new cluster points average
        for clusterIndex in range(0, self.amountOfClusters):                        
            clusterVector = self.getClusterVector(clusterIndex)                     # Gets the cluster vector i.e. the vector beloning to the cluster index that contains all the ID's of the points that are in that cluster.
            clusterVectorSize = self.getClusterVectorSize(clusterVector)            
            sumOfClusterVectorEntries = self.calculateSumOfClusterVectorEntriesNonStandardized(clusterVector)
            self.setCentroidOfCluster(clusterIndex, clusterVectorSize, sumOfClusterVectorEntries)  # calculate and set the new centroid
    
    #############################################################################
    # DIFFERENT!
    def calculateSumOfClusterVectorEntriesNonStandardized(self, clusterVector):                    # Calculate the sum of all points in the given clusterVector
        sum = np.zeros(self.amountOfColumns - 1)
        for ID in clusterVector:
            pointIndex = np.where(self.idVector == ID)[0][0]
            nonStandardizedPoint = self.originalDataWithoudID[pointIndex, :]
            sum += nonStandardizedPoint
        return sum
    
    # def setCentroidsCorrect(self):
    #     for clusterIndex in range(0, self.amountOfClusters):
    #          clusterVector = self.getClusterVector(clusterIndex)
    #          clusterVectorSize = 
    #          sumOfClusterEntries = self.calculateSumOfOriginalClusterVectorEntries()
    #          centroidVector = calculateNewCentroid()
    #          for ID in clusterVector:
    #              pointIndex = np.where(self.idVector == ID)[0][0]
    #              point = self.originalData[pointIndex, :]
            
    #########################################################################################################
    #  Composite funcitions
    #########################################################################################################
        
    def firstIteration(self):                                   # Only called for first iteration, sets the first centroids by means of the maxima of the data columns.
        self.standardizeData()
        self.kMeansPlusPlusMethod()                             # Set start centroids by K++    
        self.setDistanceOfPointsToCentroidsMatrix()             # Get distance points to centroids matrix
        self.setClusterDictionaryFirstRun()                     # Set cluster dictionary
        
    def improveLossFunctionValueKernel(self):                   # Is called in every loop to decrease the Loss function Value by resetting the centroids in a better wat
        self.setKAccentValues()
        self.setCentroids()
        self.setClusterDictionary()
        
        
