import scipy.linalg 
import numpy as np
import pandas as pd


def arnoldi_iteration(A, b, n: int):
    """Computes a basis of the (n + 1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^n b}.

    Arguments
      A: m × m array
      b: initial vector (length m)
      n: dimension of Krylov subspace, must be >= 1
    
    Returns
      Q: m x (n + 1) array, the columns are an orthonormal basis of the
        Krylov subspace.
      h: (n + 1) x n array, A on basis Q. It is upper Hessenberg.  
    """
    eps = 1e-12
    h = np.zeros((n+1,n))
    Q = np.zeros((A.shape[0],n+1))
     # Normalize the input vector
    Q[:,0] =b/np.linalg.norm(b,2)   # Use it as the first Krylov vector
    for k in range(1,n+1):
        v = np.dot(A,Q[:,k-1])      # Generate a new candidate vector
        for j in range(k):          # Subtract the projections on previous vectors
            h[j,k-1] = np.dot(Q[:,j].T, v)
            v = v - h[j,k-1] * Q[:,j]
        h[k,k-1] = np.linalg.norm(v,2)
        if h[k,k-1] > eps:  # Add the produced vector to the list, unless
            Q[:,k] = v/h[k,k-1]
        else:  # If that happens, stop iterating.
            return Q, h
    return Q, h


def QR_Decomposition(A):
    n, m = A.shape # get the shape of A

    Q = np.empty((n, n)) # initialize matrix Q
    u = np.empty((n, n)) # initialize matrix u

    u[:, 0] = A[:, 0]
    Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

    for i in range(1, n):

        u[:, i] = A[:, i]
        for j in range(i):
            u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j] # get each u vector

        Q[:, i] = u[:, i] / np.linalg.norm(u[:, i]) # compute each e vetor

    R = np.zeros((n, m))
    for i in range(n):
        for j in range(i, m):
            R[i, j] = A[:, j] @ Q[:, i]

    return Q, R


def QR_eigvals(A, tol=1e-15, maxiter=1000):
    "Find the eigenvalues of A using QR decomposition."

    A_old = np.copy(A)
    A_new = np.copy(A)

    diff = np.inf
    i = 0
    while (diff > tol) and (i < maxiter):
        A_old[:, :] = A_new
        Q, R = QR_Decomposition(A_old)

        A_new[:, :] = R @ Q

        diff = np.abs(A_new - A_old).max()
        i += 1

    eigvals = np.diag(A_new)

    return eigvals

#variables
Laplacian = pd.read_csv("Dataset/Laplacian.csv").to_numpy()
A = Laplacian
K = 8
sample_size = A.shape[0]

#set starting vector for arnoldi iteration
b = np.zeros(sample_size)

#arnoldi iteration and remove last row, this is not needed
Q, h = arnoldi_iteration(A, b, K)
h = h[:K, :]

#print(scipy.linalg.eig(h))
eigenvalues = QR_eigvals(h)

#only take the K smallest eigenvalues
K_smallest_eigenvalues = np.argsort(eigenvalues)[:K]

#print the eigenvectors
for ev in K_smallest_eigenvalues:
    print(ev)
    print(scipy.linalg.null_space((A - ev*np.eye(sample_size))))
    print("\n")


#correctEigenvalues,  correctEigenvectors = scipy.linalg.eig(A)
#print(correctEigenvectors)
#print("\n")


# correctEV1 = np.array([ 0.40824829, -0.80178373,  0.35052374])
# correctEV2 = np.array([-0.40824829, -0.26726124,  0.93472998])
       
# calculatedEV1 = np.array([-0.81649658, -0.40824829, 0.40824829])
# calculatedEV2 = np.array([ 0.53452248, -0.80178373, -0.26726124])


