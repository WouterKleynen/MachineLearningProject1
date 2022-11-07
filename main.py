import pandas as pd
import numpy as np

df = pd.read_csv('Dataset/EastWestAirlinesCluster.csv')

data = df.to_numpy()

#Size of practice data
N = 500

#Define variable for K-means clustering
K = 10

#Initialize dataset and find maxima for all columns
practice_data = data[0:N,1:]

maxima = np.zeros(11)
for i in range(11):
    maxima[i] = max(practice_data[:,i])
    
#Set centroids as  evenly spaced accross all columns
centroids = np.zeros((K,11))    
for x in range(K):
    for i in range(11):
        centroids[x,i] = int(maxima[i]*x/K)

#Define Euclidean distance

def Euclidean(a,b):
    dist = np.linalg.norm(a-b)
    return dist
    

#Compare all data to every centroid and see which is closest
distances = np.zeros((len(practice_data),len(centroids)))
for x in range (len(practice_data)):
    for i in range(len(centroids)):
        distances[x,i] = Euclidean(practice_data[x],centroids[i])
