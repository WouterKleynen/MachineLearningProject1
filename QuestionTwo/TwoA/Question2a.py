import scipy
import numpy as np
from scipy.linalg import eigh
import math
import pandas as pd


class LaplacianMatrix:
    
    def __init__(self, dataFilePath, K, sigma):
        self.sigma = sigma
        self.data = pd.read_csv(dataFilePath).to_numpy()
        self.dataWithoutIDMatrix = self.data[:, 1:]
        self.NumberOfRows = np.shape(self.dataWithoutIDMatrix)[0]
        self.NumberOfColumns = np.shape(self.dataWithoutIDMatrix)[1]
        self.amountOfClusters = K
        self.KSmallestEigenvaluesVector = np.zeros((1, self.amountOfClusters))
        self.KSmallestEigenvectorsMatrix = np.zeros((1, self.amountOfClusters))
        self.Laplacian = np.zeros((self.NumberOfColumns, self.NumberOfColumns))

    def getKSmallestEigenvectorsAndValues(self):
        KsmallestEigenvaluesAndVectors = eigh(self.Laplacian, eigvals= (0, self.amountOfClusters))
        self.KSmallestEigenvaluesVector = KsmallestEigenvaluesAndVectors[0]
        self.KSmallestEigenvectorsMatrix = KsmallestEigenvaluesAndVectors[1]
        
    # Gets Euclidean distance of 2 vectors 
    def getEuclideanDistance(self, a,b):
        return np.linalg.norm(a-b)

    def getGaussianDistance(self, point1, point2, sigma):
        return math.exp(-self.getEuclideanDistance(point1, point2)/(2 * sigma^2))

    def fillMatrixW(self):
        for point in range(0, self.NumberOfRows):
            for pointDeeper in range(0, self.NumberOfRows):
                self.Laplacian[point][pointDeeper] = self.getGaussianDistance(point, pointDeeper, self.sigma)
        print(self.Laplacian)
        

# Create a CSV file for this specific K value that will contain the eventual clusters. First emptry.
def createCSVClusterFilesKernel(K):
    for clusterIndex in range (0,K):
        euclideanClusteredCSVFile = pd.DataFrame(columns=['ID#','Balance','Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles', 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll','Award?'])
        euclideanClusteredCSVFile.to_csv(f'Dataset\KernelClusteredData\Cluster{clusterIndex}.csv', index=False) # No index used
        
                
dataFilePath = "Dataset/testing.csv"
LaplacianMatrix(dataFilePath, 10, 10).fillMatrixW()