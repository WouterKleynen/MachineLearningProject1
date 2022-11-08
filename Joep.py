#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:01:42 2022

@author: jawclaes
"""
import math
import pandas as pd
import numpy as np

df = pd.read_csv('EastWestAirlinesCluster.csv')

data = df.to_numpy()

#Size of practice data
N = 100


#Define variable for K-means  clustering
K = 10

#Initialize dataset and find maxima for all columns
practice_labels = data[0:N,0]
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
    

#Calculate distance between each data point and centroid
distances = np.zeros((len(practice_data),len(centroids)))
for x in range (len(practice_data)):
    for i in range(len(centroids)):
        distances[x,i] = Euclidean(practice_data[x],centroids[i])

#Find closest centroid for all data points
closest = np.zeros(N)
compare = np.zeros(K)
for i in range(N):
    compare = distances[i,:]
    mymin = np.min(compare)
    min_positions = [a for a, x in enumerate(compare) if x == mymin]
    closest[i] = int(min_positions[0])

#Add ID of all data points in same cluster to list
clusters = np.zeros((K,N))
for i in range(N):
    print(int(closest[i]),practice_labels[i])
    first_zero = np.where(clusters[int(closest[i])] == 0)[0]
    clusters[int(closest[i]),first_zero[0]] = practice_labels[i]

#Set position of centroids equal to mean position of all data points in the cluster
for x in range(K):
    weight = 0
    total = np.zeros(0)
    for i in range(N):
        if clusters[x,i] == 0:
            break
        weight =+ 0
        total[:] =+ [a for a, b in enumerate(compare) if b == cluster]




















