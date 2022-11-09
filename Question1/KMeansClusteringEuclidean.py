import pandas as pd
import numpy as np
from Question1.Tools import *

# Throughout the code we denote point i by the row vector at position i of the dataFile, not ID i. 
# Each variable of Vector or Matrix type is denoted as such at the end of the variable name.

class KMeansClusteringEuclidean:
    
    # Setup for the start parameters
    def __init__(self, dataFilePath, K):
        self.data                            = pd.read_csv(dataFilePath).to_numpy()
        self.amountOfClusters                = K
        self.amountOfRows                    = len(self.data)
        self.amountOfColumns                 = len(self.data[0]) # Since we will remove the ID's we usually work with 1 column less
        self.idVector                        = self.data[:, 0]
        self.dataWithoutIDMatrix             = self.data[:, 1:] # Remove ID's
        self.columnsMaximaVector             = np.zeros(self.amountOfColumns - 1)
        self.centroidsMatrix                 = np.zeros((self.amountOfClusters, self.amountOfColumns-1))
        self.centroidToPointsDistancesMatrix = np.zeros((self.amountOfRows, len(self.centroidsMatrix)))
        self.clusterDictionary               = {}
    
    #########################################################################################################
    # Getter functions
    #########################################################################################################
    
    # Gets maxima of each column and store these values in columnsMaximaVector
    def getMaximaOfColumns(self):
        for columnIndex in range(self.amountOfColumns-1):
            self.columnsMaximaVector[columnIndex] = max(self.dataWithoutIDMatrix[:,columnIndex])
    
    # Gets the index of the centroid which is closest to the point at rowIndex of the data.
    def getIndexClosestCentroid(self, rowIndex):
        return np.argmin(self.centroidToPointsDistancesMatrix[rowIndex])
    
    # Gets the cluster vector given the corresponding clusterIndex from the clusterDictionary by its index. 
    def getClusterVector(self, clusterIndex):
        return self.clusterDictionary[clusterIndex]
    
    # Gets the length of each cluser i.e. the amount of points it contains.
    def getClusterVectorSize(self, clusterVector):
        return len(clusterVector)
    
    # Gets the corresponding point index (row index) in dataWithoutIDMatrix given the ID value.
    def getPointIndexFromId(self, id):
        return np.where(self.idVector == id)[0][0]
    
    # Gets the corresponding point (row) in dataWithoutIDMatrix given the Point Index value. 
    def getPointFromPointIndex(self, pointIndex):
        return self.dataWithoutIDMatrix[pointIndex, :]
    
    # Get the clusterIndex-th row of the centroidsMatrix to get the centroid belonging to the cluster of that index 
    def getCentroidVector(self, clusterIndex):
        return self.centroidsMatrix[clusterIndex, :]
     
    #########################################################################################################
    # Setter functions
    #########################################################################################################
    
    # Sets the Matrix where Row i stores the distance of point i to each cluster indexed by the column index.
    def setDistanceOfPointsToCentroidsMatrix(self):
        for rowIndex in range (0, self.amountOfRows):
            for centroidIndex in range(self.amountOfClusters):
                self.centroidToPointsDistancesMatrix[rowIndex, centroidIndex] = getEuclideanDistance(self.dataWithoutIDMatrix[rowIndex], self.centroidsMatrix[centroidIndex])
    
    # Sets centroids at the start of the algorithm as evenly spaced accross all columns.
    # Returns the Matrix where row i consists of the start Centroid values for Cluster i. 
    def setStartCentroids(self):
        for clusterIndex in range(0, self.amountOfClusters):
            for columnIndex in range(0, self.amountOfColumns - 1):
                # Here we use clusterIndex + 1, otherwise the first cluster (0-th index) would cause mulitplication by 0 and thus the startCentroid would be the 0 vector. 
                self.centroidsMatrix[clusterIndex, columnIndex] = int(self.columnsMaximaVector[columnIndex] * (clusterIndex + 1) / (self.amountOfClusters)) 
    
    # Sets each cluster key of the clusterDictionary with all points that are closest to that cluster.
    def setClusterDictionary(self):
        self.emptyClusterDictionary() # empty the old cluster vectors
        for rowIndex in range(self.amountOfRows):
            id = self.idVector[rowIndex]
            closestClusterIndex = self.getIndexClosestCentroid(rowIndex)
            self.clusterDictionary[closestClusterIndex].append(id)

    # set the clusterIndex-th row of centroidsMatrix to the new centroid of that cluster        
    def setCentroidOfCluster(self, clusterIndex, clusterVectorSize, sumOfClusterVectorEntries):
        self.centroidsMatrix[clusterIndex, :] = self.calculateNewCentroid(clusterVectorSize, sumOfClusterVectorEntries)
    
    # Sets the Centroids of all clusters by calculatin the new cluster points average
    def setCentroids(self):
        for clusterIndex in range(0, self.amountOfClusters):
            clusterVector = self.getClusterVector(clusterIndex)
            clusterVectorSize = self.getClusterVectorSize(clusterVector)
            sumOfClusterVectorEntries = self.calculateSumOfClusterVectorEntries(clusterVector)
            self.setCentroidOfCluster(clusterIndex, clusterVectorSize, sumOfClusterVectorEntries)
    
    #########################################################################################################
    # Calculation functions
    #########################################################################################################

    # Calculate the sum of all points in the given clusterVector
    def calculateSumOfClusterVectorEntries(self, clusterVector):
        sum = np.zeros(self.amountOfColumns - 1)
        for id in clusterVector:
            sum += self.getPointFromPointIndex(self.getPointIndexFromId(id))    
        return sum
                
    # Calculate the new averaged value of the centroid of the given cluster. If a cluster has no points then return a vector with only 0 as an entry
    def calculateNewCentroid(self, clusterVectorSize, sumOfClusterVectorEntries):
        if (clusterVectorSize == 0): 
            return np.zeros(1)
        else:
            return sumOfClusterVectorEntries / clusterVectorSize
        
    # Calculate the sum of distances of the data points to the centers of the clusters they belong to
    def calculateLossFunctionValue(self):
        loss = 0
        for clusterIndex in range(0, self.amountOfClusters):
            clusterVector = self.getClusterVector(clusterIndex)
            centroidVector = self.getCentroidVector(clusterIndex)
            for id in clusterVector:
                point = self.getPointFromPointIndex(self.getPointIndexFromId(id))    
                loss += getEuclideanDistance(centroidVector, point)
        return loss
    
    #########################################################################################################
    # General functions
    #########################################################################################################

    # Empties the dictionary that contains clusterIndeces as keys and all points that are closest to that cluster as its entry        
    def emptyClusterDictionary(self):
        for clusterIndex in range(0, self.amountOfClusters):
            self.clusterDictionary[clusterIndex] = []
    
    #########################################################################################################
    #  Functions used in Question1.py
    #########################################################################################################

    # Sets the first centroids by means of the maxima of the data columns
    def firstIteration(self):
        self.getMaximaOfColumns()
        self.setClusterDictionary()
        self.setStartCentroids()

    # Is called in every loop to decrease the Loss function Value by resetting the centroids in a better wat
    def improveLossFunctionValue(self):
        self.setDistanceOfPointsToCentroidsMatrix()
        self.setClusterDictionary()
        self.setCentroids()




