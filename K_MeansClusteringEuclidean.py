import pandas as pd
import numpy as np


class K_MeansClusteringEuclidean:
    
    def __init__(self, dataFilePath, K):
        self.data           = pd.read_csv(dataFilePath).to_numpy()
        self.amountOfClusters = K
        self.amountOfRows = len(self.data)
        self.M              = len(self.data[0])
        self.practice_data  = self.data[0:self.amountOfRows,1:] # remove ID's
        self.maxima         = np.zeros(self.M-1)
        self.centroids      = np.zeros((self.amountOfClusters-1,self.M-1))
        self.distances      = np.zeros((len(self.practice_data), len(self.centroids)))

        
    #Initialize dataset and find maxima for all columns
    def getMaximaColumns(self):
        for i in range(self.M-1):
            self.maxima[i] = max(self.practice_data[:,i])
    
    #Set centroids as evenly spaced accross all columns
    def updateCentroids(self):
        for x in range(self.amountOfClusters - 1):
            for i in range(self.M-1):
                self.centroids[x,i] = int(self.maxima[i]*x/(self.amountOfClusters-1)) 
        
    #Define Euclidean distance
    def Euclidean(self, a,b):
        dist = np.linalg.norm(a-b)
        return dist
    
    #Compare all data to every centroid and see which is closest
    def getDistances(self):
        for x in range (len(self.practice_data)):
            for i in range(len(self.centroids)):
                self.distances[x,i] = self.Euclidean(self.practice_data[x], self.centroids[i])
                


# classInstance = K_MeansClusteringEuclidean('Dataset/EastWestAirlinesCluster.csv', 10)
# classInstance.getMaximaColumns()
# classInstance.updateCentroids()
# classInstance.getDistances()
# print(classInstance.distances)

# Old code:

# df = pd.read_csv('Dataset/EastWestAirlinesCluster.csv')
# data = df.to_numpy()

# #Size of practice data
# N = len(data)
# M = len(data[0])

# #Define variable for K-means clustering
# K = 9

# #Initialize dataset and find maxima for all columns
# practice_data = data[0:N,1:]
# maxima = np.zeros(M-1)
# for i in range(M-1):
#     maxima[i] = max(practice_data[:,i])
        
# #Set centroids as evenly spaced accross all columns
# centroids = np.zeros((K,M-1))    
# for x in range(K):
#     for i in range(M-1):
#         centroids[x,i] = int(maxima[i]*x/K)    

# #Define Euclidean distance
# def Euclidean(a,b):
#     dist = np.linalg.norm(a-b)
#     return dist

# #Compare all data to every centroid and see which is closest
# distances = np.zeros((len(practice_data), len(centroids)))
# for x in range (len(practice_data)):
#     for i in range(len(centroids)):
#         distances[x,i] = Euclidean(practice_data[x], centroids[i])
        
# print(distances)