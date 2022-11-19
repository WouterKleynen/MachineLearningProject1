import numpy as np
import pandas as pd
import math

# Gets Euclidean distance of 2 vectors 
def getEuclideanDistance(a,b):
    return np.linalg.norm(a-b)

# Gets Gaussian distance of 2 vectors given a sigma
def getGaussianDistance(point1, point2, sigma):
    return math.exp(-(getEuclideanDistance(point1, point2)/(2 * sigma**2))**2)

# Create a CSV file for this specific K value that will contain the eventual clusters. First emptry.
def createCSVClusterFilesKernel(K):
    for clusterIndex in range (0,K):
        euclideanClusteredCSVFile = pd.DataFrame(columns=['ID#','Balance','Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles', 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll','Award?'])
        euclideanClusteredCSVFile.to_csv(f'Dataset\KernelClusteredData\Cluster{clusterIndex}.csv', index=False) # No index used
        