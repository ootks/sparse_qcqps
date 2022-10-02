# Sparse QCQPs
This repository implements the algorithms described in https://arxiv.org/abs/2208.11143.
This includes C++ implementations of the sparse linear regression and sparse PCA tasks and ipython notebooks for experiments testing the performance of these methods.
The C++ implementation comes with python bindings by default.
There is also a python implementation of an earlier (slower) vserion of these algorithms.

Sparse linear regression is formally defined as follows:
```
    min ||A x - b||_2^2 : ||x||_0 <= k
```

Sparse PCA is formally defined as follows:
```
    min xAx : ||x||_0 <= k, ||x|| = 1.
```

## Installing
To run the code in this repository, you will need the [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) C++ package, which can be installed by downloading the Eigen package and adding it to your system's path, or by installing it using your operating system's package manager.

To install the python bindings for the C++ implementation, move to this directory and run
```bash
pip install ./lpm_methods
```

To run ipython notebooks in this repository, you will need to install the following python packages:
* numpy
* sklearn
## Usage
After intalling, import this package by running
```python
import lpm_methods
```

For sparse linear regression, try
```python
import numpy as np
A = np.random.normal(loc = 0, scale = 1, size = (10,10))
b = np.random.normal(loc = 0, scale = 1, size = (10))
T = lpm_methods.regression(A, b, 2)
```
T will then be a list of integers. These integers represent a set of 2 columns of A that have small the least squared error when you regress b against these columns.

For sparse PCA, try
```python
import numpy as np
A = np.random.normal(loc = 0, scale = 1, size = (10,10))
T = lpm_methods.pca(A, 2)
```
T will then be a list of integers. These integers represent a set of 2 columns of A that have large maximum eigenvalue.
