import pandas as pd
import numpy as np
import scipy
from KMeansClustering import KMeansClustering

class NEW(KMeansClustering):
    
    # Setup for the start parameters
    def __init__(self, dataFilePath, K, kernel):
        
        super(NEW, self).__init__(dataFilePath, K)
        # The inherited self.dataWithoutIDMatrix is set to standardized data in the first step of the firstiteration: self.standardizeData()
        
        self.kernel                          = kernel                       # Kernel function that is being used
        self.nonStandardizedData             = self.data                    # Nonstandardized data with ID
        self.nonStandardizedDataWithoutID    = self.dataWithoutIDMatrix     # Nonstandardized data without ID
        self.standardizedData                = np.array((self.amountOfRows, self.amountOfColumns))  # Standardized data
        self.standardizedDataWithoutID       = np.array((self.amountOfRows, self.amountOfColumns - 1))  # Standardized data without ID
        self.kAccentMatrix                   = np.zeros((self.amountOfRows, self.amountOfClusters)) # Row i consists of the distances of the kAccent values of point i.
        
    
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
