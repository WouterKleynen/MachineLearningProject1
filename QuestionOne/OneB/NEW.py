import pandas as pd
import numpy as np
from Tools import getGaussianDistance
from KMeansClustering import KMeansClustering


class KernelKMeansClustering(KMeansClustering):
    
    # Setup for the start parameters
    def __init__(self, sigma):
        self.sigma                           = sigma
        self.kAccentMatrix                   = np.zeros((self.amountOfRows, self.amountOfClusters))           # Row i consists of the distances of point i to each cluster.                                                            # Each entry consists of a key that's the cluster index and a value that's a vector containing all the ID's of the points that belong to that cluster.

    #########################################################################################################
    # Getter functions
    #########################################################################################################

    def getIndexMinimumCluster(self, rowIndex):                                    # Given row index i, it gets the index of the minimum of row i of kDistanceMatrix
        return np.argmin(self.kAccentMatrix[rowIndex])
    
    #########################################################################################################
    # Setter functions
    #########################################################################################################

    def setKAccentValuesNew(self):
        clusterVectorSizes = self.getClusterVectorSizesVector()
        K3Vector           = self.sumOfGaussianDistanceWithAllPointsVector()
        for rowIndex in range (0, self.amountOfRows):
            for clusterIndex in range(self.amountOfClusters):
                self.kAccentMatrix[rowIndex, clusterIndex] = self.getKAccentValueNew(self.dataWithoutIDMatrix[rowIndex], clusterIndex, K3Vector[clusterIndex], clusterVectorSizes[clusterIndex])

    def setClusterDictionaryFirstRun(self):                                        # For each key (clusterIndex) in the clusterDictionary, determines which points are closests to the centroid of that cluster, then it adds the ID's of these points to the clusterVector being the value belonging to the clusterIndex key.
        self.emptyClusterDictionary()                                              # empty the old cluster vectors.
        for rowIndex in range(self.amountOfRows):                                  # iterate over all the points.
            id = self.idVector[rowIndex]                                           # Get the ID belonging to each point.
            closestClusterIndex = self.getIndexClosestCentroid(rowIndex)           # Get the index of closest centroid by finding the minimum of row i of centroidToPointsDistancesMatrix.
            self.clusterDictionary[closestClusterIndex].append(id)
    
    #########################################################################################################
    # Calculation functions
    #########################################################################################################

    def sumOfGaussianDistanceWithPoint(self, point, clusterIndex):
        totalSum = 0
        clusterVector = self.getClusterVector(clusterIndex)
        for ID in clusterVector:
            pointInCluster = self.getPointFromID(ID)           
            totalSum       += getGaussianDistance(point, pointInCluster, self.sigma)
        return totalSum
    
    def sumOfGaussianDistanceWithAllPoints(self, clusterIndex):
        totalSum = 0
        clusterVector = self.getClusterVector(clusterIndex)
        for ID in clusterVector:
            pointInCluster = self.getPointFromID(ID)
            for IDAgain in clusterVector:
                pointInclusterAgain = self.getPointFromID(IDAgain)
                totalSum            += getGaussianDistance(pointInCluster, pointInclusterAgain, self.sigma)
        return totalSum

    def sumOfGaussianDistanceWithAllPointsVector(self):
        sumOfGaussianDistanceWithAllPointsVector = []
        for clusterIndex in range(0, self.amountOfClusters):
            value = self.sumOfGaussianDistanceWithAllPoints(clusterIndex)
            sumOfGaussianDistanceWithAllPointsVector.append(value)
        return sumOfGaussianDistanceWithAllPointsVector
    
    def getKAccentValueNew(self, point, clusterIndex, sumOfGaussianDistanceWithAllPoints, clusterSize):
        if clusterSize == 0:
            return None
        firstTerm       = 1
        sumOfGaussianDistanceWithPoint     = self.sumOfGaussianDistanceWithPoint(point, clusterIndex)
        secondTerm = (- 2.0 / clusterSize) * sumOfGaussianDistanceWithPoint
        thirdTerm = (1.0 / clusterSize**2) * sumOfGaussianDistanceWithAllPoints
        value = firstTerm + secondTerm + thirdTerm
        return value
            
    #########################################################################################################
    #  Composite funcitions
    #########################################################################################################
        
    def firstIteration(self):                                   # Only called for first iteration, sets the first centroids by means of the maxima of the data columns.
        self.kMeansPlusPlusMethod()                             # Set start centroids by K++    
        self.setDistanceOfPointsToCentroidsMatrix()             # Get distance points to centroids matrix
        self.setClusterDictionaryFirstRun()                     # Set cluster dictionary
        
    def improveLossFunctionValueKernel(self):                   # Is called in every loop to decrease the Loss function Value by resetting the centroids in a better wat
        self.setKAccentValuesNew()
        self.setClusterDictionary()


