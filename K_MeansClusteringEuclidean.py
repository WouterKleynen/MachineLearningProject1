import pandas as pd
import numpy as np


class K_MeansClusteringEuclidean:
    
    # Setup of the start parameters
    def __init__(self, dataFilePath, K):
        self.data                      = pd.read_csv(dataFilePath).to_numpy()
        self.amountOfClusters          = K
        self.amountOfRows              = len(self.data)
        self.amountOfColumns           = len(self.data[0]) # Since we will remove the ID's we usually work with 1 column less
        self.idArray                   = self.data[:, 0]
        self.practice_data             = self.data[:, 1:] # Remove ID's
        self.columnsMaximaVector       = np.zeros(self.amountOfColumns - 1)
        self.centroids                 = np.zeros((self.amountOfClusters, self.amountOfColumns-1))
        self.centroidToPointsDistances = np.zeros((self.amountOfRows, len(self.centroids)))
        self.pointToClusterDictionary  = {} 
        
    # Gets maxima for all columns
    def getMaximaOfColumns(self):
        for columnIndex in range(self.amountOfColumns-1):
            self.columnsMaximaVector[columnIndex] = max(self.practice_data[:,columnIndex])
    
    # Set centroids at the start of the algorithm as evenly spaced accross all columns
    def setStartCentroids(self):
        for clusterIndex in range(self.amountOfClusters - 1):
            for columnIndex in range(self.amountOfColumns - 1):
                self.centroids[clusterIndex, columnIndex] = int(self.columnsMaximaVector[columnIndex] * clusterIndex / (self.amountOfClusters)) 
        
    #Define Euclidean distance of 2 vectors 
    def getEuclideanDistance(self, a,b):
        return np.linalg.norm(a-b)
        
    # Gets the distance of every point to each centroid. Row i stores the distance of data row i (point i) to each cluster
    def getDistanceOfPointsToCentroids(self):
        for rowIndex in range (self.amountOfRows):
            for centroidIndex in range(len(self.centroids)):
                self.centroidToPointsDistances[rowIndex, centroidIndex] = self.getEuclideanDistance(self.practice_data[rowIndex], self.centroids[centroidIndex])
    
    # Gets the index of the cluster of which it's centroid is closest to data row i
    def getIndecesClosestCentroids(self):
        indecesClosestCentroids = []
        for rowIndex in range(self.amountOfRows):
            indecesClosestCentroids.append(np.argmin(self.centroidToPointsDistances[rowIndex]))
        return indecesClosestCentroids
    
    # Gets the minimum of the row of each Centroid to Points distance matrix, i.e. the index of the closest cluster to each point
    def getMinimumOfCentroidToPointsDistancesRow(self, rowIndex):
        return np.argmin(self.centroidToPointsDistances[rowIndex])
    
    # Updates dictionary where each key is the ID and the value is the closest cluster index 
    def updatePointToClusterDictionary(self):
        for rowIndex in range(self.amountOfRows):
            self.pointToClusterDictionary[self.idArray[rowIndex]] =  self.getMinimumOfCentroidToPointsDistancesRow(rowIndex)
            



# import pandas as pd
# import numpy as np

# df = pd.read_csv('Dataset/EastWestAirlinesCluster.csv')

# data = df.to_numpy()

# #Size of practice data
# N = len(data)
# M = len(data[0])

# #Define variable for K-means clustering
# K = 10

# #Initialize dataset and find maxima for all columns
# practice_data = data[0:N,1:]

# maxima = np.zeros(11)
# for i in range(11):
#     maxima[i] = max(practice_data[:,i])
    
# #Set centroids as  evenly spaced accross all columns
# centroids = np.zeros((K,11))    
# for x in range(K):
#     for i in range(11):
#         centroids[x,i] = int(maxima[i]*x/K)

# #Define Euclidean distance

# def Euclidean(a,b):
#     dist = np.linalg.norm(a-b)
#     return dist
    

# #Compare all data to every centroid and see which is closest
# distances = np.zeros((len(practice_data),len(centroids)))
# for x in range (len(practice_data)):
#     for i in range(len(centroids)):
#         distances[x,i] = Euclidean(practice_data[x],centroids[i])

# print(distances)