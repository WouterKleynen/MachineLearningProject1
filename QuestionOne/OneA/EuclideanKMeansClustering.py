import pandas as pd
import numpy as np
from KMeansClustering import KMeansClustering

class EuclideanKMeansClustering(KMeansClustering):
     
    #########################################################################################################
    # Setter functions
    #########################################################################################################
    
    def setRandomStartCentroids(self):                                              # Picks uniformly chosen random centroids out of the data set for the first iteration
        min_, max_ = np.min(self.dataWithoutIDMatrix, axis=0), np.max(self.dataWithoutIDMatrix, axis=0)
        self.centroidsMatrix = np.array([np.random.uniform(min_, max_) for _ in range(self.amountOfClusters)]) 

    def setCentroidOfCluster(self, clusterIndex, clusterVectorSize, sumOfClusterVectorEntries):         # Calculate new centroid based on the points in the cluster and set this new centroid in centroidsMatrix at the clusterIndex row.
        self.centroidsMatrix[clusterIndex, :] = self.calculateNewCentroid(clusterVectorSize, sumOfClusterVectorEntries)
    
    def setCentroids(self):                                                         # Sets the Centroids of all clusters by calculatin the new cluster points average
        for clusterIndex in range(0, self.amountOfClusters):                        
            clusterVector = self.getClusterVector(clusterIndex)                     # Gets the cluster vector i.e. the vector beloning to the cluster index that contains all the ID's of the points that are in that cluster.
            clusterVectorSize = self.getClusterVectorSize(clusterVector)            
            sumOfClusterVectorEntries = self.calculateSumOfClusterVectorEntries(clusterVector)
            self.setCentroidOfCluster(clusterIndex, clusterVectorSize, sumOfClusterVectorEntries)  # calculate and set the new centroid
    
    #########################################################################################################
    # Calculation functions
    #########################################################################################################

    def calculateSumOfClusterVectorEntries(self, clusterVector):                    # Calculate the sum of all points in the given clusterVector
        sum = np.zeros(self.amountOfColumns - 1)
        for id in clusterVector:
            sum += self.getPointFromID(id)    
        return sum
                
    def calculateNewCentroid(self, clusterVectorSize, sumOfClusterVectorEntries):   # Calculate the new averaged value of the centroid of the given cluster. 
        if (clusterVectorSize == 0):                                                # If a cluster has no ID's then return a vector with only 0 as an entry
            return np.zeros(1)
        else:
            return sumOfClusterVectorEntries / clusterVectorSize
        
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
    #  Composite funcitions
    #########################################################################################################

    def firstIteration(self):                                   # Only called for first iteration, sets the first centroids by means of the maxima of the data columns.
        self.kMeansPlusPlusMethod()                             # Set start centroids by K++    
        self.setDistanceOfPointsToCentroidsMatrix()             # Get distance points to centroids matrix
        self.setClusterDictionary()                             # Set cluster dictionary
        self.setCentroids()                                     # Set new centroids

    def firstIterationRandom(self):                             # Only called for first iteration, sets the first centroids by means of the maxima of the data columns.
        self.setRandomStartCentroids()                          # Set start centroids by K++    
        self.setDistanceOfPointsToCentroidsMatrix()             # Get distance points to centroids matrix
        self.setClusterDictionary()                             # Set cluster dictionary
        self.setCentroids()                                     # Set new centroids

    def improveLossFunctionValue(self):                           # Is called in every loop to decrease the Loss function Value by resetting the centroids in a better wat
        self.setDistanceOfPointsToCentroidsMatrix()
        self.setClusterDictionary()
        self.setCentroids()
        
# Gets Euclidean distance of 2 vectors 
def getEuclideanDistance(a,b):
    return np.linalg.norm(a-b)

# Create a CSV file for this specific K value that will contain the eventual clusters. First emptry.
def createCSVClusterFiles(K):
    for clusterIndex in range (0,K):
        euclideanClusteredCSVFile = pd.DataFrame(columns=['ID#','Balance','Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles', 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll','Award?'])
        euclideanClusteredCSVFile.to_csv(f'Dataset\EuclideanClusteredData\Cluster{clusterIndex}.csv', index=False) # No index used


