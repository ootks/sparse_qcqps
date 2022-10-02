#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace poly {
    using Eigen::VectorXd;
    double esp(VectorXd v, uint k);
}
