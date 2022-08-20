"""Computes characteristic coefficients"""

import numpy as np
# Computes an elementary symmetric polynomial of degree k in a vector x.
def esp(x, k):
    if k == 0:
        return 1
    if k == 1:
        return sum(x)
    n = len(x)
    S = np.zeros((n+1, k))
    for j in range(1, n+1):
        S[j, 0] = S[j-1, 0] + x[j-1]
    for i in range(1, k):
        for j in range(1, n+1):
            S[j, i] = S[j-1, i] + x[j-1] * S[j-1, i-1]
    return S[n, k-1]

# Computes characteristic polynomials as an elementary symmetric polynomial in
# the eigenvalues of X.
def char_coeff_eigen(X, k):
    return esp(np.linalg.eigvalsh(X), k)

def char_coeff(X,k):
    return char_coeff_eigen(X,k)

# Swaps ith and jth rows and columns of X.
def swap(X, i, j):
    if i == j:
        return
    for k in range(len(X)):
        X[k,i], X[k,j]  = X[k,j], X[k,i]
    for k in range(len(X)):
        X[i,k], X[j,k]  = X[j,k], X[i,k]

# Computes the conditional polynomial p|_[t], where p is the characteristic
# coefficient of X in degree k
def conditional_char(X, t, k):
    schur = X[t:, t:] - X[t:, :t] @ np.linalg.inv(X[:t, :t]) @ X[:t, t:]
    return np.linalg.det(X[:t, :t]) * char_coeff(schur, k-t)

