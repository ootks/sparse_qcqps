#include "poly.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Tridiagonalization;
using Eigen::SelfAdjointEigenSolver;

namespace poly {
/**
 * Helper function for computing degree k characteristic coefficient of a 
 * tridiagonal matrix.
 *
 * @param diagonal Size n vector containing diagonal of the matrix.
 * @param subdiagonal Size n-1 vector containing subdiagonal of the matrix.
 * @param k Degree of the polynomial.
 * @return degree k characteristic coefficient of corresponding tridiagonal
 * matrix.
 */
double char_coeff_help(VectorXd diagonal, VectorXd subDiagonal, uint k)
{
    size_t n = diagonal.rows();
    double** S = new double*[n+1];
    for (uint i = 0; i <= n; i++) {
        S[i] = new double[3];
        S[i][0] = 1;
    }
    S[0][1] = 0;
    for (uint i = 1; i <= n; i++) {
        S[i][1] = S[i-1][1] + diagonal(i-1);
    }
    for (uint j = 2; j <= k; j++) {
        uint index = j%3;
        uint prev_index = (j-1)%3;
        uint ante_index = (j-2)%3;
        for (uint i = 0; i < j; i++) {
            S[i][j] = 0;
        }
        for (uint i = j; i <= n; i++) {
            double prev = S[i-1][index];
            double diff = diagonal(i-1) * S[i-1][prev_index];
            double entry = subDiagonal(i-2);
            double correction = entry * entry * S[i-2][ante_index];
            S[i][index] = prev + diff - correction;
        }
    }
    double answer = S[n][k%3];

    for (uint i = 0; i <= n; i++) {
        delete[] S[i];
    }
    delete[] S;

    return answer;
}

/**
 * Computes the degree k characteristic coefficient of a tridiagonal
 * matrix.
 *
 * @param diagonal Size n vector containing diagonal of the matrix.
 * @param subdiagonal Size n-1 vector containing subdiagonal of the matrix.
 * @param k Degree of the polynomial.
 * @return degree k characteristic coefficient of corresponding tridiagonal
 * matrix.
 */
double char_coeff_tri(VectorXd diagonal, VectorXd subDiagonal, uint k) {
    if (k == 0) {
        return 1;
    }
    if (k == 1) {
        return diagonal.sum();
    }
    return char_coeff_help(diagonal, subDiagonal, k);
}

/**
 * Computes the degree k characteristic coefficient of a matrix with * given
 * tridiagonalizer.
 *
 * @param X An nxn matrix.
 * @param k Degree of the polynomial.
 * @param tri Eigen tridiagonalizer.
 * @return degree k characteristic coefficient of X.
 */
double char_coeff(MatrixXd X, uint k, Tridiagonalization<MatrixXd>* tri)
{
    tri->compute(X);
    return char_coeff_tri(tri->diagonal(), tri->subDiagonal(), k);
}

/**
 * Computes the degree k characteristic coefficient of a matrix.
 *
 * @param X An nxn matrix.
 * @param k Degree of the polynomial.
 * @return degree k characteristic coefficient  of X.
 */
double char_coeff(MatrixXd X, uint k)
{
    Tridiagonalization<MatrixXd> tri;
    return char_coeff(X, k, &tri);
}


/**
 * Computes the degree k elementary symmetric polynomial of a vector.
 *
 * @param v A vector.
 * @param k Degree of the polynomial.
 * @return degree k elementary symmetric polynomial of X.
 */
double esp(VectorXd v, uint k)
{
    if (k == 0) {
        return 1;
    }
    if (k == 1) {
        return v.sum();
    }

    size_t n = v.rows();

    double** S = new double*[n+1];
    for (uint i = 0; i <= n; i++) {
        S[i] = new double[2];
    }
    S[0][1] = 0;

    for (uint i = 1; i <= n; i++) {
        S[i][1] = S[i-1][1] + v(i-1);
    }
    S[0][0] = 0;
    for (uint j = 2; j <= k; j++) {
        for (uint i = 1; i <= n; i++) {
            S[i][j%2] = v(i-1) * S[i-1][(j-1)%2] + S[i-1][j%2];
        }
    }
    double answer = S[n][k%2];
    for (uint i = 0; i <= n; i++) {
        delete[] S[i];
    }
    delete[] S;
    return answer;
}

/**
 * Computes the degree k characteristic coefficient of a matrix.
 *
 * @param X An nxn matrix.
 * @param k Degree of the polynomial.
 * @return degree k characteristic coefficient of X.
 */
double char_coeff_old(MatrixXd X, uint k)
{
    SelfAdjointEigenSolver<MatrixXd> es;
    return esp(es.compute(X).eigenvalues(), k);
}

}
