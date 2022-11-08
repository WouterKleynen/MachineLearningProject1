import pandas as pd
import numpy as np

# Throughout the code we denote point i by the row vector at position i of the dataFile, not ID i. 
# Each variable of Vector or Matrix type is denoted as such at the end of the variable name.

class K_MeansClusteringEuclidean:
    
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
        # create dictionary that contains clusterIndeces as keys and all points that are closest to that cluster as its entry
        for clusterIndex in range(0, self.amountOfClusters):
            self.clusterDictionary[clusterIndex] = []
            
        
    # Gets maxima of each column and store these values in columnsMaximaVector
    def getMaximaOfColumns(self):
        for columnIndex in range(self.amountOfColumns-1):
            self.columnsMaximaVector[columnIndex] = max(self.dataWithoutIDMatrix[:,columnIndex])
    
    # Sets centroids at the start of the algorithm as evenly spaced accross all columns.
    # Returns centroidsMatrix where row i consists of the start Centroid values for Cluster i. 
    def setStartCentroids(self):
        for clusterIndex in range(self.amountOfClusters - 1):
            for columnIndex in range(self.amountOfColumns - 1):
                self.centroidsMatrix[clusterIndex, columnIndex] = int(self.columnsMaximaVector[columnIndex] * clusterIndex / (self.amountOfClusters)) 
        
    # Gets Euclidean distance of 2 vectors 
    def getEuclideanDistance(self, a,b):
        return np.linalg.norm(a-b)
        
    # Gets the distance of every point to each centroid. 
    # Returns a centroidToPointsDistancesMatrix where Row i stores the distance of point i to each cluster.
    def getDistanceOfPointsToCentroids(self):
        for rowIndex in range (self.amountOfRows):
            for centroidIndex in range(len(self.centroidsMatrix)):
                self.centroidToPointsDistancesMatrix[rowIndex, centroidIndex] = self.getEuclideanDistance(self.dataWithoutIDMatrix[rowIndex], self.centroidsMatrix[centroidIndex])
    
    # # Gets the index of the cluster of which it's centroid is closest to data row i
    # def getIndecesClosestCentroids(self):
    #     indecesClosestCentroids = []
    #     for rowIndex in range(self.amountOfRows):
    #         indecesClosestCentroids.append(np.argmin(self.centroidToPointsDistancesMatrix[rowIndex]))
    #     return indecesClosestCentroids
    
    
    # Gets the index of the cluster which is closest to the point at rowIndex of the data.
    def getIndexClosestCluster(self, rowIndex):
        return np.argmin(self.centroidToPointsDistancesMatrix[rowIndex])
    
    # Sets each cluster key of the clusterDictionary with all points that are closest to that cluster.
    def setClusterDictionary(self):
        for rowIndex in range(self.amountOfRows):
            id = self.idVector[rowIndex]
            closestClusterIndex = self.getIndexClosestCluster(rowIndex)
            self.clusterDictionary[closestClusterIndex].append(id)
    
    # Gets the cluster entry from the clusterDictionary by its index. 
    def getClusterEntries(self, clusterIndex):
        return self.clusterDictionary[clusterIndex]
    
    # Gets the length of each cluser i.e. the amount of points it contains.
    def getClusterSize(self, clusterVector):
        return len(clusterVector)
    
    # def getPointFromId(self, id):
    #     index = np.where(matrix[0, :] == value)[0].item()
    #     return 
    
    # def getSumOfClusterEntries(self, clusterVector):
    #     return sum(clusterVector)
    
    # def updateCenters(self):
    #     for clusterIndex in range(0, self.amountOfClusters):
    #         IDsInCluster = self.getClusterEntries(clusterIndex)
    #         sumOfClusterEntries = self.getSumOfClusterEntries
    #         return 
    