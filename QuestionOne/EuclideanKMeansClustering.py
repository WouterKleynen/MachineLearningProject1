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
        
    #########################################################################################################
    # Calculation functions
    #########################################################################################################

        

    
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

    def improveLossFunctionValue(self):                         # Is called in every loop to decrease the Loss function Value by resetting the centroids in a better wat
        self.setDistanceOfPointsToCentroidsMatrix()
        self.setClusterDictionary()
        self.setCentroids()
        
    # Gets Euclidean distance of 2 vectors 
    def getEuclideanDistance(self, a,b):
        return np.linalg.norm(a-b)

    # Create a CSV file for this specific K value that will contain the eventual clusters. First emptry.
    def createCSVClusterFiles(self):
        for clusterIndex in range (0,self.amountOfClusters):
            euclideanClusteredCSVFile = pd.DataFrame(columns=['ID#','Balance','Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles', 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll','Award?'])
            euclideanClusteredCSVFile.to_csv(f'Dataset\EuclideanClusteredData\Cluster{clusterIndex}.csv', index=False) # No index used


