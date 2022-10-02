# Sparse QCQPs
This repository implements the algorithms described in https://arxiv.org/abs/2208.11143.
This includes C++ implementations of the sparse linear regression and sparse PCA tasks and ipython notebooks for experiments testing the performance of these methods.
The C++ implementation comes with python bindings by default.
There is also a python implementation of an earlier (slower) vserion of these algorithms.

Sparse linear regression is formally defined as follows:
$$
    \min \{ \|A x - b\|_2^2 : \|\supp(x)\| \le k\}.
$$

Sparse PCA is formally defined as follows:
$$
\begin{equation*}
\begin{aligned}
    \max\quad & x^{\intercal}Ax\\
    \st & x^{\intercal}x = 1\\
        & x \in \R^n\\
        &|\supp(x)| \le k.
\end{aligned}
\end{equation*}
$$

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
```python
pip install ./lpm_methods
```
