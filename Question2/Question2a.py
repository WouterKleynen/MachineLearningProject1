import scipy
import numpy
from scipy.linalg import eigh


class LaplacianMatrix():
    
    def __init__(self, K):
        self.Laplacian = numpy.array([[1,1,1], [2,2,2], [3,3,3]]) #Change to actual Laplacian matrix
        self.NumberOfRows = numpy.shape(self.Laplacian)[0]
        self.NumberOfColumns = numpy.shape(self.Laplacian)[1]
        self.K = K
        self.KSmallestEigenvaluesVector = numpy.zeros((1, self.K))
        self.KSmallestEigenvectorsMatrix = numpy.zeros((1, self.K))

def getKSmallestEigenvectorsAndValues(self):
    KsmallestEigenvaluesAndVectors = eigh(self.Laplacian, eigvals= (0, self.K))
    self.KSmallestEigenvaluesVector = KsmallestEigenvaluesAndVectors[0]
    self.KSmallestEigenvectorsMatrix = KsmallestEigenvaluesAndVectors[1]