import scipy.linalg
import numpy as np
import pandas as pd
from math import sqrt

#This method only gives the smallest eigenvalue and eigenvector, not the K smallest values and vectors

def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)

def power_iteration(A, max_error = 10**(-10), max_iteration = 1000):
    n, d = A.shape
    i = 0
    error = 100
    ev = np.ones(d) / np.sqrt(d)
    lambda_old = eigenvalue(A, ev)

    while error > max_error and i < max_iteration:
        i += 1
        Av = A.dot(ev)
        ev_new = Av / np.linalg.norm(Av)

        lambda_new = eigenvalue(A, ev_new)
        error = np.abs(lambda_old - lambda_new)
        lambda_old = lambda_new
        ev = ev_new
    return lambda_new, ev_new

#variables
Laplacian = pd.read_csv("Dataset/Laplacian.csv").to_numpy()
A = Laplacian
K = 8
sample_size = A.shape[0]
max_error = 10**(-2)
max_iterations = 5000


#return largest eigenvector
lambda_largest, EV_largest = power_iteration(A, max_error, max_iterations)

#return smallest eigenvalue and eigenvector by doing the power iteration method on A - lambda_largest * I
B = A - lambda_largest * np.eye(sample_size)
lambda_smallest, EV_smallest = power_iteration(B, 10**(-3), 10000)
lambda_smallest = lambda_smallest + lambda_largest


##check whether our eigenvalues are close
#correctEigenvalues,  correctEigenvectors = scipy.linalg.eig(A)
