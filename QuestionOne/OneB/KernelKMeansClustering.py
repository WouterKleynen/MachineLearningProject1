import pandas as pd
import numpy as np
from KMeansClustering import KMeansClustering

class KernelKMeansClustering(KMeansClustering):
    
    # Setup for the start parameters
    def __init__(self, dataFilePath, K, kernel):        
        super(KernelKMeansClustering, self).__init__(dataFilePath, K)             # Inherit from KernelKMeansClustering
        self.kernel = kernel                      
        self.kAccentMatrix = np.zeros((self.amountOfRows, self.amountOfClusters)) # Row i consists of the kAccent values of point i.
        self.IDClusterDictionary = {}                                             # Dictionary where each key is an ID and the value is the cluster it belongs to.
        self.T3Vector = np.zeros(self.amountOfClusters)                           # Vector that contains the values of the double sum of the third term from the k accent calculation, 
        self.counter = 0
        
    def standardize(self, columnIndex):                                                         # Basic standardization                           
        mu = np.average(self.data[:, columnIndex])
        sigma = np.std(self.data[:, columnIndex])
        Z = (self.data[:, columnIndex] - mu)/sigma
        return Z
    
    def standardizeData(self):                                                                  # Standardized the input data and saves the standardized data in a CSV
        standardizedMatrix = np.zeros((self.amountOfRows, self.amountOfColumns))
        for columnIndex in range(0, self.amountOfColumns):
            standardizedMatrix[:, columnIndex] = self.standardize(columnIndex)                  # Standardize each column
        self.dataWithoutIDMatrix = standardizedMatrix[:, 1:]                                    # Set the columns of matrix that's in use for input data: dataWithoutIDMatrix, to the standardized column of the input data
        pd.DataFrame(self.dataWithoutIDMatrix).to_csv("Dataset/standardizedData.csv",index=False, header=False) # Write standardized data to a CSV

    def getIndexMinimumCluster(self, rowIndex):                                                 # Given row index i, gets the index of the minimum of row i of kAccentMatrix
        return np.argmin(self.kAccentMatrix[rowIndex])
    
    def setKAccentValues(self):
            clusterVectorSizes = self.getClusterVectorSizesVector()                         # Store the amount of IDs that each vector has
            for rowIndex in range (0, self.amountOfRows):                                   # Iterate over the data points
                for clusterIndex in range(self.amountOfClusters):                           # Iterate over the clusters
                    point        = self.dataWithoutIDMatrix[rowIndex]                       # get normalized point
                    T3Value      = self.T3Vector[clusterIndex]                              # Get the value of double sum as it's stored in the T3 vector
                    clusterSize  = clusterVectorSizes[clusterIndex]                         # Get the clustersize
                    self.kAccentMatrix[rowIndex, clusterIndex] = self.getKAccentValue(point, clusterIndex, T3Value, clusterSize)
                print(f"k accent values determined for point {rowIndex}")
    
    def setFirstClusterDictionary(self):                                           # For each key (clusterIndex) in the clusterDictionary, determines which points are closests to the centroid of that cluster, then it adds the ID's of these points to the clusterVector being the value belonging to the clusterIndex key.
        self.emptyClusterDictionary()                                              # empty the old cluster vectors.
        for rowIndex in range(self.amountOfRows):                                  # iterate over all the points.
            id = self.idVector[rowIndex]                                           # Get the ID belonging to each point.
            closestClusterIndex = self.getIndexClosestCentroid(rowIndex)           # Get the index of closest centroid by finding the minimum of row i of centroidToPointsDistancesMatrix.
            self.clusterDictionary[closestClusterIndex].append(id)                 # Add that index to the dictionary
            self.IDClusterDictionary[id] = closestClusterIndex                     # Set the value of cluster index to the key ID
        print("Calculating all T3 values started")
        self.T3Vector = self.CalculateT3ValuesVector()                                   # Calculate all T3 values, this has to be done only once at the beginning of the run of the algorithm
        print("Calculating all T3 values finished")
        
    def constructEmptyDictionary(self):                                            # Constructs an empty dictionary with amountOfClusters keys and values
        clusterDictionary = {}
        for clusterIndex in range(0, self.amountOfClusters):
            clusterDictionary[clusterIndex] = []
        return clusterDictionary

    def setKernelClusterDictionary(self):   
        print("start of dictionary")
        clusterDictionary = self.constructEmptyDictionary()
        for rowIndex in range(self.amountOfRows):                                   # iterate over all the points.
            id                          = self.idVector[rowIndex]
            point                       = self.getPointFromID(id)
            previousClosestClusterIndex = self.IDClusterDictionary[id]     
            newClosestClusterIndex      = self.getIndexMinimumCluster(rowIndex)     # Get the index of the minimum entry of row i, that index is the cluster index of the cluster that point i is going to be put in.
            if (newClosestClusterIndex != previousClosestClusterIndex):             # Compare new cluster Index to the old 
                 previousClusterT3 = self.T3Vector[previousClosestClusterIndex]     # Get the T3 value from old cluster
                 newClusterT3      = self.T3Vector[newClosestClusterIndex]          # Get the T3 value from new cluster
                 decrease          = (2 * self.T3decreaseValue(id, previousClosestClusterIndex) - self.kernel(point, point))    # Calculate the value that decreases the old cluster T3 value
                 increase          = 2 * self.T3increaseValue(id, newClosestClusterIndex) - self.kernel(point, point)           # Calculate the value that increases the new cluster T3 value
                 self.T3Vector[previousClosestClusterIndex] = previousClusterT3 - decrease
                 self.T3Vector[newClosestClusterIndex]      = newClusterT3 + increase
                 self.IDClusterDictionary[id]               = newClosestClusterIndex  # Set the ID to the new cluster
            clusterDictionary[newClosestClusterIndex].append(id)                      # Add the new ID to the corresponding cluster vector in the cluster dictionary
        self.clusterDictionary = clusterDictionary                                    # Once the full dictionary is constructed, set the self to it
        print("end of dictionary")
            
    def T3increaseValue(self, id, clusterIndex):
        totalSum = 0
        clusterVector = self.getClusterVector(clusterIndex)
        clusterVector.append(id)                                                    # add the id to the clusterVector of the transferred cluster
        for ID in clusterVector:                                                    # Loop over the IDs in the vector
            pointInCluster = self.getPointFromID(ID)                                # gets point from nonStandardizedIDVector
            point = self.getPointFromID(id)                                         
            totalSum += self.kernel(point, pointInCluster)                          # calculate the total increase
        return totalSum
    
    def T3decreaseValue(self, id, clusterIndex):
        totalSum = 0                                                                # do not remove the id to the clusterVector of the transferred cluster
        clusterVector = self.getClusterVector(clusterIndex)                         
        for ID in clusterVector:
            pointInCluster = self.getPointFromID(ID)                                # gets point from nonStandardizedIDVector
            point = self.getPointFromID(id)
            totalSum += self.kernel(point, pointInCluster)                          # calculate the total increase
        return totalSum
        
    def CalculateT3ValuesVector(self):
        sumOfKernelOfAllPointsInClusterVector = []
        for clusterIndex in range(0, self.amountOfClusters):                        # Iterate over the clusters
            value = self.CalculateT3Value(clusterIndex)              # Calculate the T3 per cluster
            sumOfKernelOfAllPointsInClusterVector.append(value)
            print(f"T3 for clusterIndex: {clusterIndex} calculated")
        return sumOfKernelOfAllPointsInClusterVector

    def CalculateT3Value(self, clusterIndex):                        
        totalSum = 0
        clusterVector = self.getClusterVector(clusterIndex)
        for ID in clusterVector:                                                     # Loop over the ID's
            pointInCluster = self.getPointFromID(ID)
            for IDAgain in clusterVector:                                            # Second loop
                pointInclusterAgain = self.getPointFromID(IDAgain)
                totalSum += self.kernel(pointInCluster, pointInclusterAgain)         # update the total sum
        return totalSum                                                              # Returns the T3 value

    def getKAccentValue(self, point, clusterIndex, sumOfKernelOfAllPointsInCluster, clusterSize):       # Calculates the k accent value per point and cluster
        firstTerm                          = self.kernel(point, point)                                  # Calculates the first term
        sumOfGaussianDistanceWithPoint     = self.calculateSecondTermSum(point, clusterIndex)           # Calculates the sum in the second term, this part makes the whole algorithm painfully slow
        secondTerm = (- 2.0 / clusterSize) * sumOfGaussianDistanceWithPoint                             # Calculates the second term
        thirdTerm = (1.0 / clusterSize**2) * sumOfKernelOfAllPointsInCluster                            # Calculates the third term
        value = firstTerm + secondTerm + thirdTerm                                                      # Calculates k accent
        return value

    def calculateSecondTermSum(self, point, clusterIndex):                          # Calculates the second term sum 
        totalSum = 0
        clusterVector = self.getClusterVector(clusterIndex)                         
        for ID in clusterVector:                                                    # Loops over the IDs in the cluster vector
            pointInCluster = self.getPointFromID(ID)                                # Gets point from nonStandardizedIDVector
            totalSum      += self.kernel(point, pointInCluster)                     # Calculates the new term and add it the total
        return totalSum
    
    def fillClusterCSV(self):                                                       # Fill each Clusters CSV file with the points belonging to it. 
        for clusterIndexKey in self.clusterDictionary:
            clusterIDVector = self.clusterDictionary[clusterIndexKey]
            for pointID in clusterIDVector:
                dataPoint = self.getPointFromIDWithID(pointID)
                dataFrame = pd.DataFrame(dataPoint)
                dataFrame.T.to_csv(f'Dataset/KernelClusteredData{clusterIndexKey}.csv', mode='a', index=False, header=False)
    
    def firstIteration(self):                                   # First iteration is done by regular K means clustering
        print("Standardizing data started")
        self.standardizeData()                                  # All input data is standardized and saved in a CVS file
        print("Standardizing data finished")
        print("K++ started")
        self.kMeansPlusPlusMethod()                             # Set the start centroids
        print("K++ ended")
        self.setDistanceOfPointsToCentroidsMatrix()             # Set distance of points to centroids
        print("Start dictionary start")
        self.setFirstClusterDictionary()                        # Set the dictionary that contains clusterindices as keys and vectors with the corresponding ID's that it cointains as value
        print("Start dictionary finished")
        
    def improveKernelLossFunctionValue(self):                   # Keep on running this function untill the loss function stops improving
        print("Setting k accent values started")
        self.setKAccentValues()
        print("Setting k accent values finished")
        self.counter = 0
        print("set dictionary started")
        self.setKernelClusterDictionary()
        print("set dictionary finished")
