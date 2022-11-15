import pandas as pd
import numpy as np
from Tools import getEuclideanDistance, getGaussianDistance, getGaussianDistanceWithItself
import scipy
import math

# Throughout the code we denote point i by the row vector at position i of the dataFile, not ID i. 
# Each variable of Vector or Matrix type is denoted as such at the end of the variable name.

class KMeansClusteringKernel:
    
    # Setup for the start parameters
    def __init__(self, dataFilePath, K, sigma):
        self.sigma                           = sigma
        self.data                            = pd.read_csv(dataFilePath).to_numpy()
        self.amountOfClusters                = K
        self.amountOfRows                    = len(self.data)
        self.amountOfColumns                 = len(self.data[0])                                              # Since we will remove the ID's we usually work with 1 column less.
        self.idVector                        = self.data[:, 0]                                                # Only ID's are extracted.
        self.dataWithoutIDMatrix             = self.data[:, 1:]                                               # ID's are removed.
        self.startCentroidsMatrix            = np.zeros((self.amountOfClusters, self.amountOfColumns - 1))    # Row i is the centroid of cluster i. 
        self.kDistanceMatrix                 = np.zeros((self.amountOfRows, self.amountOfClusters))           # Row i consists of the distances of point i to each cluster.
        self.clusterDictionary               = {}                                                             # Each entry consists of a key that's the cluster index and a value that's a vector containing all the ID's of the points that belong to that cluster.
        
    #########################################################################################################
    # Getter functions
    #########################################################################################################
    
    def getIndexMinimumCluster(self, rowIndex):                                    # Given row index i, it gets the index of the minimum of row i of kDistanceMatrix
        return np.argmin(self.kDistanceMatrix[rowIndex])
    
    def getClusterVector(self, clusterIndex):                                       # Gets the vector containing all the ID's of the points that belong to cluster clusterIndex
        return self.clusterDictionary[clusterIndex]
    
    def getClusterVectorSize(self, clusterVector):                                  # Gets the length of each cluser i.e. the amount of (ID's) points it contains.
        return len(clusterVector)
    
    def getPointIndexFromId(self, id):                                              # Gets the row index in dataWithoutIDMatrix belonging to the given ID value.
        return np.where(self.idVector == id)[0][0]
    
    def getPointFromPointIndex(self, pointIndex):                                   # Gets the row of dataWithoutIDMatrix belonging to the given point index value. 
        return self.dataWithoutIDMatrix[pointIndex, :]
    
    def getPointFromID(self, id):                                                   # Gets the row of dataWithoutIDMatrix belonging to the given ID value. 
        return self.getPointFromPointIndex(self.getPointIndexFromId(id))
    
    def getPointFromIDWithID(self, id):                                             # Gets the row of dataWithoutIDMatrix belonging to the given ID value, with the ID included.  
        pointIndex = np.where(self.idVector == id)[0][0]
        return self.data[pointIndex, :]
        
    def getCentroidVector(self, clusterIndex):                                      # Gets the centroid belonging to cluster index.  
        return self.centroidsMatrix[clusterIndex, :]
    
    def getClusterVectorSizesVector(self):
        clusterVectorSizeVector = []
        for clusterIndex in range(0, self.amountOfClusters):
            clusterVector = self.getClusterVector(clusterIndex)                     # Gets the cluster vector i.e. the vector beloning to the cluster index that contains all the ID's of the points that are in that cluster.
            clusterVectorSize = self.getClusterVectorSize(clusterVector)
            clusterVectorSizeVector.append(clusterVectorSize)
        return clusterVectorSizeVector

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
    
    def getKAccentValue(self, point, clusterIndex):
        firstTerm       = getGaussianDistanceWithItself(point, self.sigma)
        clusterVector   = self.getClusterVector(clusterIndex)
        clusterSize     = self.getClusterVectorSize(clusterVector)
        if clusterSize == 0:
            return None
        sumOfGaussianDistanceWithPoint     = self.sumOfGaussianDistanceWithPoint(point, clusterIndex)
        sumOfGaussianDistanceWithAllPoints = self.sumOfGaussianDistanceWithAllPoints(clusterIndex)
        secondTerm = (- 2.0 / clusterSize) * sumOfGaussianDistanceWithPoint
        thirdTerm = (1.0 / clusterSize**2) * sumOfGaussianDistanceWithAllPoints
        value = firstTerm + secondTerm + thirdTerm
        return value

    #########################################################################################################
    # Setter functions
    #########################################################################################################
    
    def setKAccentValues(self):                                 
        for rowIndex in range (0, self.amountOfRows):
            for clusterIndex in range(self.amountOfClusters):
                self.kDistanceMatrix[rowIndex, clusterIndex] = self.getKAccentValue(self.dataWithoutIDMatrix[rowIndex], clusterIndex)

    def kMeansPlusPlusMethod(self):                                                 # More advanced K++ method to set the start centroids
        C = [self.dataWithoutIDMatrix[0]]                                           # Start column vector of length 11
        for _ in range(1, self.amountOfClusters):
            D2 = scipy.array([min([scipy.inner(c-x,c-x) for c in C]) for x in self.dataWithoutIDMatrix])
            probs = D2/D2.sum()
            cumprobs = probs.cumsum()
            r = scipy.rand()
            for j,p in enumerate(cumprobs):
                if r < p:
                    i = j
                    break
            C.append(self.dataWithoutIDMatrix[i])
        self.centroidsMatrix = np.array(C)
    
    def setClusterDictionary(self):   
                                                                                   # For each key (clusterIndex) in the clusterDictionary, determines which points are closests to the centroid of that cluster, then it adds the ID's of these points to the clusterVector being the value belonging to the clusterIndex key.
        self.emptyClusterDictionary()                                              # empty the old cluster vectors.
        for rowIndex in range(self.amountOfRows):                                  # iterate over all the points.
            id = self.idVector[rowIndex]                                           # Get the ID belonging to each point.
            closestClusterIndex = self.getIndexMinimumCluster(rowIndex)            # Get the index of closest centroid by finding the minimum of row i of centroidToPointsDistancesMatrix.
            self.clusterDictionary[closestClusterIndex].append(id)
    
    def calculateLossFunctionValue(self):                                           # Calculate the sum of all the distances of the data points to the centers of the clusters they belong to.        
        loss = 0
        for clusterIndex in range(0, self.amountOfClusters):
            clusterVector = self.getClusterVector(clusterIndex)
            centroidVector = self.getCentroidVector(clusterIndex)
            for id in clusterVector:
                point = self.getPointFromID(id)
                loss += getEuclideanDistance(centroidVector, point)
        return loss
    
    #########################################################################################################
    # General functions
    #########################################################################################################

    def emptyClusterDictionary(self):                                                # Empties the dictionary that contains clusterIndeces as keys and all points that are closest to that cluster as its entry        
        for clusterIndex in range(0, self.amountOfClusters):
            self.clusterDictionary[clusterIndex] = []

    def fillClusterCSV(self):                                                         # Fill each Clusters CSV file with the points belonging to it. 
        print(self.clusterDictionary)
        for clusterIndexKey in self.clusterDictionary:
            clusterIDVector = self.clusterDictionary[clusterIndexKey]
            for pointID in clusterIDVector:
                dataPoint = self.getPointFromIDWithID(pointID)
                dataFrame = pd.DataFrame(dataPoint)
                dataFrame.T.to_csv(f'Dataset\KernelClusteredData\Cluster{clusterIndexKey}.csv', mode='a', index=False, header=False)
            
    #########################################################################################################
    #  Composite funcitions
    #########################################################################################################

    def firstIteration(self):                                   # Only called for first iteration, sets the first centroids by means of the maxima of the data columns.
        self.kMeansPlusPlusMethod()                             # Set start centroids by K++    
        self.setClusterDictionary()                             # Set cluster dictionary
        self.setKAccentValues()                                 # Get distance points to centroids matrix
        
    def improveLossFunctionValueKernel(self):                   # Is called in every loop to decrease the Loss function Value by resetting the centroids in a better wat
        self.setKAccentValues()
        self.setClusterDictionary()


