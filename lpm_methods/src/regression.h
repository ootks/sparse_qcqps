#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>

namespace sreg {
    using Eigen::MatrixXd;
    using Eigen::VectorXd;
    using std::vector;


    vector<uint> regression(MatrixXd A, VectorXd b, uint k);
}
