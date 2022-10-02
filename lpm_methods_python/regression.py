"""Approximates sparse regression using characteristic coefficients"""

import numpy as np
from .char_poly import *

# Finds set T that is a set of k columns of A that minimizes the least squares
# regression error for b.
def regression(A, b, k):
    print("Starting with k=",k)
    n = A.shape[1]
    T = []
    npts = 5
    # Use chebyshev nodes
    xs = [0,1]#[np.cos((2*i-1)/(2*npts) * np.pi) for i in range(npts)]
    X0 = np.transpose(A)@A
    V0 = (np.transpose(A) @ np.outer(b, b) @ A)
    Xs = [X0 + x * V0 for x in xs]
    bests = []
    for t in range(k):
        print("Round ", t)
        best = -1
        best_heur = 0
        for j in range(t, n):
            chars = []
            for X in Xs:
                swap(X, t, j)
                chars.append(conditional_char(X/1000, t+1, k))
                swap(X, t, j)
            
            line = np.polyfit(xs, chars, 1)
            heur = line[0]/line[1]
            if heur > best_heur:
                best = j
                best_heur = heur
        for X in Xs:
            swap(X, t, best)
        try:
            while True:
                best = T.index(best)
        except ValueError:
            T.append(best)
        bests.append(best_heur - 1)
    return T, bests
