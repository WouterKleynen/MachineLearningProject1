import numpy as np
import pandas as pd

# Gets Euclidean distance of 2 vectors 
def getEuclideanDistance(a,b):
    return np.linalg.norm(a-b)

# Create a CSV file for this specific K value that will contain the eventual clusters. First emptry.
def createCSVClusterFiles(K):
    for clusterIndex in range (0,K):
        euclideanClusteredCSVFile = pd.DataFrame(columns=['ID#','Balance','Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles', 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll','Award?'])
        euclideanClusteredCSVFile.to_csv(f'Dataset\EuclideanClusteredData\Cluster{clusterIndex}.csv', index=False) # No index used
