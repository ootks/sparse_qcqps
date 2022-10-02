#define _USE_MATH_DEFINES
#include "pca.h"
#include "conditionals.h"
#include "poly.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::VectorXd;
using Eigen::EigenvaluesOnly;
using Eigen::SelfAdjointEigenSolver;
using std::vector;
using std::find;
using cond::conditionals_inc;
using poly::esp;

namespace spca {
// Computes the Schur complement of X with respect to row/column i.
void schur_complement(MatrixXd* X, uint i) {
    VectorXd row = X->row(i);
    MatrixXd update = row * row.transpose() / row(i);
    (*X) -= update;
}

// Computes coefficients for Lagrange interpolation with respect to chevychev
// nodes.
ArrayXd factors(uint dim) {
    ArrayXd coeffs(dim);
    coeffs = 1;
    double sign = 1;
    for (uint i = 1; i < dim-1; i++) {
        sign = -sign;
        coeffs(i) = sign * 0.5;
    }
    coeffs(dim-1) = -sign;
    return coeffs;
}

// Computes one newton step for the polynomial implicitly defined by its
// evaluations (given in vals), at the points ts, at the point y. coeffs
// are the coefficients for Lagrange interpolation, precomputed for efficiency.
double newton_step(ArrayXd ts, ArrayXd coeffs, ArrayXd vals, double y) {
    ArrayXd inverses = 1/(y - ts);
    ArrayXd map = coeffs * inverses;
    double eval = (map * vals).sum();
    map *= (inverses.sum() - inverses);
    double derivative = (map * vals).sum();

    return eval / derivative;
}

// Computes the maximum root of the polynomial implicitly defined by its values
// (given in vals) at the points given in ts. coeffs are the coefficients for 
// Lagrange interpolation, precomputed for efficiency.
double max_root(ArrayXd ts, ArrayXd coeffs, VectorXd vals) {
    // Assumes that the maximum root is at most 1.
    double y = 1;

    while (true) {
        double step = newton_step(ts, coeffs, vals, y);
        if (step < 1e-7) {
            break;
        }
        y -= step;
    }

    return y;
}

// Computes the matrix of the form aA + bI whose maximum eigenvalue is max_eig,
// and whose minimum eigenvalue is min_eig.
void normalize_matrix(MatrixXd* A, SelfAdjointEigenSolver<MatrixXd>* es,
                      double max_eig, double min_eig) {
    VectorXd eigvals = es->compute(*A, EigenvaluesOnly).eigenvalues();
    double scale = (max_eig - min_eig) /
                   (eigvals.maxCoeff() - eigvals.minCoeff());
    double shift = min_eig - scale * eigvals.minCoeff();
    *A *= scale;
    *A += shift * MatrixXd::Identity(A->cols(), A->cols());
}

// Finds a list of k indices of A so that the principal submatrix of A 
// corresponding to those indices maximizes the maximum eigenvalue out of all
// principal submatrices of A with k indices.
vector<uint> pca(MatrixXd A, uint k) {
    vector<uint> output;

    size_t n = A.cols();
    uint dim = k+1;
    SelfAdjointEigenSolver<MatrixXd> es;
    // Rescale A to make sure that its eigenvalues are within [1/2, 1].
    normalize_matrix(&A, &es, 1, 0.5);

    // This needs to be updated
    ArrayXd eval_pts(dim);
    MatrixXd* matrix_eval_pts = new MatrixXd[dim];
    ArrayXd dets(dim);
    // We should evaluate the polynomial at chbyshev nodes
    for (uint i = 0; i < dim; i ++) {
        double pt =  (cos(i * M_PI / (dim-1)) + 2.9)/4;
        eval_pts(i) = pt;
        matrix_eval_pts[i] = pt * MatrixXd::Identity(n, n) - A;
        dets(i) = 1;
    }

    ArrayXd coeffs = factors(dim);

    // Scores the conditional evaluations for each index i, and each eval point
    ArrayXXd conditionals(n, dim);

    for (uint t = 0; t < k; t++) {
        // Evaluate all of the conditionals for each evaluation point.
        for (uint i = 0; i < dim; i ++) {
            VectorXd cond = conditionals_inc(matrix_eval_pts[i], &es, k-t);
            conditionals.col(i) = cond;
        }

        double biggest = -1;
        uint best = 0;
        bool found_candidate = false;
        for (uint i = 0; i < n; i++) {
            if (std::count(output.begin(), output.end(), i) != 0) {
                continue;
            }
            VectorXd evals = dets * conditionals.row(i).transpose();
            double val = max_root(eval_pts, coeffs, evals);
            if (val > biggest) {
                biggest = val;
                best = i;
                found_candidate = true;
            }
        }
        assert(found_candidate);
        for (uint i = 0; i < dim; i++) {
            dets(i) *= matrix_eval_pts[i](best,best);
            schur_complement(&matrix_eval_pts[i], best);
        }
        output.push_back(best);
    }
    delete[] matrix_eval_pts;
    return output;
}
}
