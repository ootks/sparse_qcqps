#include "regression.h"
#include "conditionals.h"
#include "poly.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Tridiagonalization;
using Eigen::SelfAdjointEigenSolver;
using std::vector;
using std::find;
using cond::conditionals_inc;
using poly::esp;

namespace sreg {
void schur_complement(MatrixXd* X, uint i) {
    VectorXd row = X->row(i);
    MatrixXd update = row * row.transpose() / row(i);
    (*X) -= update;
}

vector<uint> regression(MatrixXd A, VectorXd b, uint k) {
    size_t n = A.cols();
    MatrixXd X = A.transpose() * A / 10;
    VectorXd c = A.transpose() * b;
    MatrixXd Z = X +  c * c.transpose() / 10;
    vector<uint> output;
    SelfAdjointEigenSolver<MatrixXd> es;
    for (uint t = 0; t < k; t++) {
        VectorXd p_X = conditionals_inc(X, &es, k-t);
        VectorXd p_Z = conditionals_inc(Z, &es, k-t);
        double biggest = -1;
        uint best = 0;
        bool found_candidate = false;
        for (uint i = 0; i < n; i++) {
            if (X(i,i) == 0 || Z(i,i) == 0 ||
                    std::count(output.begin(), output.end(), i) != 0) {
                continue;
            }
            double val = p_Z(i) / p_X(i);
            if (val > biggest) {
                biggest = val;
                best = i;
                found_candidate = true;
            }
        }
        if (!found_candidate) {
            throw std::runtime_error("No good index found. "
                                     "This is likely due to a numerical error."
                                     );
        }
        schur_complement(&X, best);
        schur_complement(&Z, best);
        output.push_back(best);
    }
    return output;
}
}
