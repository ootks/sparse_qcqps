#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>

namespace spca {
    using Eigen::MatrixXd;
    using std::vector;

    vector<uint> pca(MatrixXd A, uint k);
}
