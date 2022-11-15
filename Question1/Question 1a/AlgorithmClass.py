import pandas as pd
import numpy as np
from Tools import getEuclideanDistance
import scipy

# Throughout the code we denote point i by the row vector at position i of the dataFile, not ID i. 
# Each variable of Vector or Matrix type is denoted as such at the end of the variable name.

class KMeansClusteringEuclidean:
    
    # Setup for the start parameters
    def __init__(self, dataFilePath, K):
        self.data                            = pd.read_csv(dataFilePath).to_numpy()
        self.amountOfClusters                = K
        self.amountOfRows                    = len(self.data)
        self.amountOfColumns                 = len(self.data[0])                                              # Since we will remove the ID's we usually work with 1 column less
        self.idVector                        = self.data[:, 0]                                                # Only ID's are extracted
        self.dataWithoutIDMatrix             = self.data[:, 1:]                                               # ID's are removed
        self.centroidsMatrix                 = np.zeros((self.amountOfClusters, self.amountOfColumns-1))      # Consists of the centroids vector where column i is centroid i. 
        self.centroidToPointsDistancesMatrix = np.zeros((self.amountOfRows, len(self.centroidsMatrix)))       # Column i consits of the distances of all points to cluster i
        self.clusterDictionary               = {}                                                             # Each entry consists of a key that's the cluster index and a value that's a vector containing all the ID's of the points that belong to that cluster
    
    #########################################################################################################
    # Getter functions
    #########################################################################################################
    
    def getIndexClosestCentroid(self, rowIndex):                                    # Gets the index of the centroid which is closest to the point, where the point is at the rowIndex of the data.
        return np.argmin(self.centroidToPointsDistancesMatrix[rowIndex])
    
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
     
    #########################################################################################################
    # Setter functions
    #########################################################################################################
    
    def setDistanceOfPointsToCentroidsMatrix(self):                                 # Sets the centroidToPointsDistancesMatrix (N x K) entries, where row i stores the distance of point i to each cluster. Or similarly where column j stores the distance of all points to cluster j.
        for rowIndex in range (0, self.amountOfRows):
            for centroidIndex in range(self.amountOfClusters):
                self.centroidToPointsDistancesMatrix[rowIndex, centroidIndex] = getEuclideanDistance(self.dataWithoutIDMatrix[rowIndex], self.centroidsMatrix[centroidIndex])
    
    def setRandomStartCentroids(self):                                              # Picks uniformly chosen random centroids out of the data set for the first iteration
        min_, max_ = np.min(self.dataWithoutIDMatrix, axis=0), np.max(self.dataWithoutIDMatrix, axis=0)
        self.centroidsMatrix = np.array([np.random.uniform(min_, max_) for _ in range(self.amountOfClusters)]) 
        
    def kMeansPlusPlusMethod(self):                                                 # More advanced K++ method to set the start centroids
        C = [self.dataWithoutIDMatrix[0]]
        for _ in range(0, self.amountOfClusters):
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
    
    def setClusterDictionary(self):                                                # For each key (clusterIndex) in the clusterDictionary, determines which points are closests to the centroid of that cluster, then it adds the ID's of these points to the clusterVector being the value belonging to the clusterIndex key.
        self.emptyClusterDictionary() # empty the old cluster vectors
        for rowIndex in range(self.amountOfRows):
            id = self.idVector[rowIndex]
            closestClusterIndex = self.getIndexClosestCentroid(rowIndex)
            self.clusterDictionary[closestClusterIndex].append(id)

    def setCentroidOfCluster(self, clusterIndex, clusterVectorSize, sumOfClusterVectorEntries):         # Calculate new centroid based on the points in the cluster and set this new centroid in centroidsMatrix at the clusterIndex row
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
            sum += self.getPointFromID(id)    
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
                point = self.getPointFromID(id)
                loss += getEuclideanDistance(centroidVector, point)
        return loss
    
    #########################################################################################################
    # General functions
    #########################################################################################################

    # Empties the dictionary that contains clusterIndeces as keys and all points that are closest to that cluster as its entry        
    def emptyClusterDictionary(self):
        for clusterIndex in range(0, self.amountOfClusters):
            self.clusterDictionary[clusterIndex] = []

    # Fill each Clusters CSV file with the points belonging to it. 
    def fillClusterCSV(self):
        for clusterIndexKey in self.clusterDictionary:
            clusterIDVector = self.clusterDictionary[clusterIndexKey]
            for pointID in clusterIDVector:
                dataPoint = self.getPointFromIDWithID(pointID)
                dataFrame = pd.DataFrame(dataPoint)
                dataFrame.T.to_csv(f'Dataset\EuclideanClusteredData\Cluster{clusterIndexKey}.csv', mode='a', index=False, header=False)
            
    #########################################################################################################
    #  Composite funcitions
    #########################################################################################################

    # Sets the first centroids by means of the maxima of the data columns
    def firstIteration(self):
        self.kMeansPlusPlusMethod()
        self.setDistanceOfPointsToCentroidsMatrix()
        self.setClusterDictionary()
        self.setCentroids()
        
    def runFirstIterationKPlusPlus(self):
        self.kMeansPlusPlusMethod()
        self.setDistanceOfPointsToCentroidsMatrix()
        self.setClusterDictionary()
        self.setCentroids()
        
    def runFirstIterationRandom(self):
        self.setRandomStartCentroids()
        self.setDistanceOfPointsToCentroidsMatrix()
        self.setClusterDictionary()
        self.setCentroids()

    # Is called in every loop to decrease the Loss function Value by resetting the centroids in a better wat
    def improveLossFunctionValue(self):
        self.setDistanceOfPointsToCentroidsMatrix()
        self.setClusterDictionary()
        self.setCentroids()
        

