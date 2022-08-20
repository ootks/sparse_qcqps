"""Approximates sparse PCA using characteristic coefficients"""

import numpy as np
from .char_poly import *

# Performs Newton's method to find the maximum root of the polynomial p.
# Assumes that the maximum root of p is at most 2.
def newton_method(p, max_iters=3000):
    # start at a point that is larger than the maximum root of p
    x = 2
    d = len(p)-1
    dpdx = [(d - i) * p[i] for i in range(d)]
    
    iters = 0
    while abs(np.polyval(p, x)) > 1e-3:
        x -= np.polyval(p, x) / np.polyval(dpdx, x)
        iters += 1
        if iters > max_iters:
            print("Newton's method failed")
            break
    return x

# Finds the maximum root of p|_[t](xI - X), where p is a characteristic
# coefficient of degree k
def root_heur(X, t, k, redund=2, D = None):
    if D is None:
        D = np.eye(len(X))
    npts = redund*k+0
    # Use chebyshev nodes centered around .75
    xs = [0.25*np.cos((2*i-1)/(2*npts) * np.pi) + 0.75 for i in range(npts)]
    # Evaluate p(X + t I) in k places
    vals = [conditional_char(x*np.eye(len(X)) - X, t+1, k) for x in xs]
    # Find the coefficients of p(X+tI)
    p = np.polyfit(xs, vals, k)
    # Use Newtons' method to find maximal root
    return newton_method(p)

# Finds set T that is the support of a vector that
# approximates the maximum k-sparse eigenvalue of A.
def pca(A, k, redund=2):
    n = A.shape[1]
    T = []
    X = A.copy()
    #Normalize X so that its eigenvalues lie in the range [1/2, 1]
    eigs = np.linalg.eigvalsh(X)
    max_eig = max(eigs)
    min_eig = min(eigs)
    scal = 1/(2*(max_eig - min_eig))
    X = scal * X + (0.5 - min_eig * scal) * np.eye(n)

    for t in range(k):
        best = -1
        best_heur = 0
        for j in range(t, n):
            swap(X, t, j)
            heur = root_heur(X, t, k, redund = redund)

            if heur > best_heur:
                best = j
                best_heur = heur
            swap(X, t, j)
        swap(X, t, best)
        try:
            while True:
                best = T.index(best)
        except ValueError:
            T.append(best)
    return T
