# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 21:11:12 2022

@author: omril
"""
import pandas as pd
import numpy as np
import scipy
import os

a = os.getcwd()
#os.chdir('Documents\GitHub\MachineLearningProject1')
#b = pd.read_csv('Dataset/InputData.csv').to_numpy()
#c = np.delete(b, np.s_[1], axis=1) 

def RemoveFeature(dataFilePath, feature):
    data = pd.read_csv(dataFilePath).to_numpy()
    dataMinusFeature = np.delete(data, np.s_[feature], axis=1)
    dataMinusFeature = pd.DataFrame(dataMinusFeature)
    dataMinusFeature.to_csv(f'QuestionThree\DataWithoutFeatures\MissingFeature' + str(feature) + '.csv', mode='a', index=False, header=False)
 
  
class KMeansClustering:
    
    # Setup for the start parameters
    def __init__(self, dataFilePath, K):
        self.data                            = pd.read_csv(dataFilePath).to_numpy()
        self.wholeData                       = pd.read_csv('Dataset/InputData.csv').to_numpy()                # The original data without missing features
        self.amountOfClusters                = K
        self.amountOfRows                    = len(self.data)
        self.amountOfColumns                 = len(self.data[0])                                              # Since we will remove the ID's we usually work with 1 column less.
        self.wholeAmountOfColumns            = len(self.wholeData[0])                                         # Same for data without missing features.
        self.idVector                        = self.data[:, 0]                                                # Only ID's are extracted.
        self.dataWithoutIDMatrix             = self.data[:, 1:]                                               # ID's are removed.
        self.wholeDataWithoutIDMatrix        = self.wholeData[:, 1:]                                          # ID's are removed from data without missing features            
        self.centroidsMatrix                 = np.zeros((self.amountOfClusters, self.amountOfColumns - 1))    # Row i is the centroid of cluster i. 
        self.centroidToPointsDistancesMatrix = np.zeros((self.amountOfRows, self.amountOfClusters))           # Row i consists of the distances of point i to each cluster.
        self.clusterDictionary               = {}                                                             # Each entry consists of a key that's the cluster index and a value that's a vector containing all the ID's of the points that belong to that cluster.
        
    #########################################################################################################
    # Getter functions
    #########################################################################################################
    
    def getIndexClosestCentroid(self, rowIndex):                                    # Given row index i, it gets the index of the minimum of row i of centroidToPointsDistancesMatrix, meaning the index of the cluster thats closest to point i. 
        return np.argmin(self.centroidToPointsDistancesMatrix[rowIndex])
    
    def getClusterVector(self, clusterIndex):                                       # Gets the vector containing all the ID's of the points that belong to cluster clusterIndex
        return self.clusterDictionary[clusterIndex]
    
    def getClusterVectorSize(self, clusterVector):                                  # Gets the length of each cluser i.e. the amount of (ID's) points it contains.
        return len(clusterVector)
    
    def getPointIndexFromId(self, id):                                              # Gets the row index in dataWithoutIDMatrix belonging to the given ID value.
        return np.where(self.idVector == id)[0][0]
    
    def getPointFromPointIndex(self, pointIndex):                                   # Gets the row of dataWithoutIDMatrix belonging to the given point index value. 
        return self.dataWithoutIDMatrix[pointIndex, :]
        
    def getFinalPointFromPointIndex(self, pointIndex):                              # Same for data without missing features.  
        return self.wholeDataWithoutIDMatrix[pointIndex, :]
    
    def getPointFromID(self, id):                                                   # Gets the row of dataWithoutIDMatrix belonging to the given ID value. 
        return self.getPointFromPointIndex(self.getPointIndexFromId(id))
    
    def getFinalPointFromID(self, id):                                              # Same for data without missing features.  
        return self.getFinalPointFromPointIndex(self.getPointIndexFromId(id))
    
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
     
    #########################################################################################################
    # Setter functions
    #########################################################################################################
    
    def setDistanceOfPointsToCentroidsMatrix(self):                                 # Sets the centroidToPointsDistancesMatrix (N x K) entries, where row i stores the distance of point i to each cluster. Or similarly where column j stores the distance of all points to cluster j.
        for rowIndex in range (0, self.amountOfRows):
            for centroidIndex in range(self.amountOfClusters):
                self.centroidToPointsDistancesMatrix[rowIndex, centroidIndex] = self.getEuclideanDistance(self.dataWithoutIDMatrix[rowIndex], self.centroidsMatrix[centroidIndex])
    
    # Picks Random Centroids first the first iteration from the data set         
    def setRandomStartCentroids(self):
        min_, max_ = np.min(self.dataWithoutIDMatrix, axis=0), np.max(self.dataWithoutIDMatrix, axis=0)
        self.centroidsMatrix = np.array([np.random.uniform(min_, max_) for _ in range(self.amountOfClusters)]) 
    
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
    
    def setClusterDictionary(self):                                                # For each key (clusterIndex) in the clusterDictionary, determines which points are closests to the centroid of that cluster, then it adds the ID's of these points to the clusterVector being the value belonging to the clusterIndex key.
        self.emptyClusterDictionary()                                              # empty the old cluster vectors.
        for rowIndex in range(self.amountOfRows):                                  # iterate over all the points.
            id = self.idVector[rowIndex]                                           # Get the ID belonging to each point.
            closestClusterIndex = self.getIndexClosestCentroid(rowIndex)           # Get the index of closest centroid by finding the minimum of row i of centroidToPointsDistancesMatrix.
            self.clusterDictionary[closestClusterIndex].append(id)
    
    def setCentroidOfCluster(self, clusterIndex, clusterVectorSize, sumOfClusterVectorEntries):         # Calculate new centroid based on the points in the cluster and set this new centroid in centroidsMatrix at the clusterIndex row.
        # print(clusterIndex)
        # print(clusterVectorSize)
        # print(sumOfClusterVectorEntries)
        # print("\n")
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
                loss += self.getEuclideanDistance(centroidVector, point) ** 2
        return loss
    
    #########################################################################################################
    # General functions
    #########################################################################################################

    def emptyClusterDictionary(self):                                                # Empties the dictionary that contains clusterIndeces as keys and all points that are closest to that cluster as its entry        
        for clusterIndex in range(0, self.amountOfClusters):
            self.clusterDictionary[clusterIndex] = []

    def fillClusterCSV(self):                                                         # Fill each Clusters CSV file with the points belonging to it. 
        for clusterIndexKey in self.clusterDictionary:
            clusterIDVector = self.clusterDictionary[clusterIndexKey]
            for pointID in clusterIDVector:
                dataPoint = self.getPointFromIDWithID(pointID)
                dataFrame = pd.DataFrame(dataPoint)
                dataFrame.T.to_csv(f'Dataset\EuclideanClusteredData\Cluster{clusterIndexKey}.csv', mode='a', index=False, header=False)
        
    # Gets Euclidean distance of 2 vectors 
    def getEuclideanDistance(self,a,b):
        return np.linalg.norm(a-b)

    # Create a CSV file for this specific K value that will contain the eventual clusters. First emptry.
    def createCSVClusterFiles(self, K):
        for clusterIndex in range (0,K):
            euclideanClusteredCSVFile = pd.DataFrame(columns=['ID#','Balance','Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles', 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll','Award?'])
            euclideanClusteredCSVFile.to_csv(f'Dataset\EuclideanClusteredData\Cluster{clusterIndex}.csv', index=False) # No index used
            
            
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

    def calculateFinalSumOfClusterVectorEntries(self, clusterVector):                # Calculate the sum of all points in the given clusterVector for the final loss function.
        sum = np.zeros(self.wholeAmountOfColumns - 1)
        for id in clusterVector:
            sum += self.getFinalPointFromID(id)    
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
                loss += getEuclideanDistance(centroidVector, point) ** 2
        return loss
    
    def calculateFinalLossFunctionValue(self):                                      # Calculate the sum of all the distances of the data points to the centers of the clusters they belong to for the final loss function.         
        loss = 0
        for clusterIndex in range(0, self.amountOfClusters):
            clusterVector = self.getClusterVector(clusterIndex)
            clusterVectorSize = self.getClusterVectorSize(clusterVector)            
            sumOfClusterVectorEntries = self.calculateFinalSumOfClusterVectorEntries(clusterVector)   #calculates the sum of all the cluster vectors including the data of the missing feature
            centroidVector = sumOfClusterVectorEntries / clusterVectorSize                            #The centroid vector with the data including the missing feature is calculated.
                         
            for id in clusterVector:
                point = self.getFinalPointFromID(id)
                loss += getEuclideanDistance(centroidVector, point) ** 2
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

    def improveLossFunctionValue(self):                         # Is called in every loop to decrease the Loss function Value by resetting the centroids in a better wat
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


# This file contains the functions that are used to iteratively call the KMeansClusteringEuclidean() class functions
# to run the K Means Euclidean Clustering algorithm until the given the treshold is reached

# Is called to run the first iteration. The first iteration differs from other iteration since it has to construct start centroids.
def runFirstIteration(dataSetFilePath, K):
    currentAlgorithmIterationValues = EuclideanKMeansClustering(dataSetFilePath, K)                 # Create an instance of the KMeansClusteringEuclidean class.
    currentAlgorithmIterationValues.firstIteration()                                                # Set the start Centroids and fill each cluster with its closest data points for the first run of the algorithm.
    return currentAlgorithmIterationValues

# Is called in every iteration to decrease the Loss Function. If the intermediate loss function values need to be printed, set printIntermediateLossFunctionValues to true
def runNewIteration(previousLossFunctionvalue, currentAlgorithmIterationValues, K):
    currentAlgorithmIterationValues.createCSVClusterFiles(K) 
    #print(currentAlgorithmIterationValues.getClusterVectorSizesVector())                                                                  
    # Create CSV file for each cluster
    #print(f"current loss fuction value = {previousLossFunctionvalue}")
    currentAlgorithmIterationValues.improveLossFunctionValue()                                      # Update the centroids by using the improveLossFunction() function
    newLossFunctionValue = currentAlgorithmIterationValues.calculateLossFunctionValue()             # Determine the value of the loss function after the new centroid update
    if (previousLossFunctionvalue == newLossFunctionValue):                                # Since newLossFunctionValue <= previousLossFuncitonvalue we get a decreasing number, we stop when they're very close i.e. their fraction is very small
        currentAlgorithmIterationValues.fillClusterCSV()                                            # Fill each cluster's CSV file with its datapoints
        finalLossFunctionValue = currentAlgorithmIterationValues.calculateFinalLossFunctionValue()
        print(currentAlgorithmIterationValues.getClusterVectorSizesVector())
        print(f"Final loss function value for K = {K} is {finalLossFunctionValue}")
        return None                                                                                 # Return None when the ratio is below the Treshold
    return newLossFunctionValue

# Runs the K Means Euclidean Clustering algorithm for a given K unt
def improveUntilTresholdReached(dataSetFilePath, K):
    currentAlgorithmIterationValues = runFirstIteration(dataSetFilePath, K)                         # Update to first Iteration (this differs from other iteration since it has to construct start centroids)
    lossFunctionvalue = currentAlgorithmIterationValues.calculateLossFunctionValue()                # Calculate the start loss function value after the first iteration
    while (lossFunctionvalue != None):                                                              # loop from 0 untill the iteration that the treshold is reached: when previousLossFunctionvalue == None
        lossFunctionvalue = runNewIteration(lossFunctionvalue, currentAlgorithmIterationValues, K)  # update each previous loss function value with a new improved one
    return lossFunctionvalue

dataSetFilePath = 'Dataset/InputData.csv'   # Set data File path to that of the assignment data sheet.
testData = "Dataset/subsetOfInputData.csv"

def analyseWithoutFeature(dataFilePath, feature):
    data = pd.read_csv(dataFilePath).to_numpy()
    dataMinusFeature = np.delete(data, np.s_[feature], axis=1)
    dataMinusFeature = pd.DataFrame(dataMinusFeature)
    dataMinusFeature.to_csv(f'QuestionThree\DataWithoutFeatures\MissingFeature' + str(feature) + '.csv', mode='a', index=False, header=False)
    print('Feature ' + str(feature) + ' is missing from the data')
    improveUntilTresholdReached('QuestionThree\DataWithoutFeatures\MissingFeature' + str(feature) + '.csv', 8) 
 
def analyseWithoutFeatures(dataFilePath, feature1, feature2):
    data = pd.read_csv('QuestionThree\DataWithoutFeatures\MissingFeature' + str(feature1) + '.csv').to_numpy()
    dataMinusFeature = np.delete(data, np.s_[feature2-1], axis=1)
    dataMinusFeature = pd.DataFrame(dataMinusFeature)
    dataMinusFeature.to_csv(f'QuestionThree\DataWithoutFeatures\MissingFeature' + str(feature1) + str(feature2) + '.csv', mode='a', index=False, header=False)
    print('Feature ' + str(feature1) + ' and '+ str(feature2) + ' is missing from the data')
    improveUntilTresholdReached('QuestionThree\DataWithoutFeatures\MissingFeature' + str(feature1) + str(feature2) + '.csv', 8) 
    
for i in range(1, 12):
    analyseWithoutFeature(dataSetFilePath, i)    

for i in range(3, 12):
    for j in range(i+1, 12):
        analyseWithoutFeatures(dataSetFilePath, i, j)  
    

        
            