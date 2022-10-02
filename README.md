# Sparse QCQPs
This repository implements the algorithms described in https://arxiv.org/abs/2208.11143.
This includes C++ implementations of the sparse linear regression and sparse PCA tasks and ipython notebooks for experiments testing the performance of these methods.
The C++ implementation comes with python bindings by default.
There is also a python implementation of an earlier (slower) vserion of these algorithms.

#Installing
To run the code in this repository, you will need the [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) C++ package, which can be installed by downloading the Eigen package and adding it to your system's path, or by installing it using your operating system's package manager.

To install the python bindings for the C++ implementation, move to this directory and run
```bash
pip install ./lpm_methods
```

To run ipython notebooks in this repository, you will need to install the following python packages:
* numpy
* sklearn
