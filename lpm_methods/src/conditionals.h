#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>


namespace cond{
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SelfAdjointEigenSolver;

VectorXd conditionals(MatrixXd X, uint k);
VectorXd conditionals_inc(MatrixXd X, SelfAdjointEigenSolver<MatrixXd>* es, uint k);
VectorXd conditionals_eigen(VectorXd diagonal, VectorXd eigenvalues, MatrixXd eigenvectors, uint k);

VectorXd gradient(VectorXd eigenvalues, uint k);
}
