#include "poly.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SelfAdjointEigenSolver;
using poly::esp;

namespace cond {
//The base 2 logarithm of n.
uint nbits(uint n) {
    uint nbits = 0;
    while (n != 0) {
        n >>= 1;
        nbits++;
    }
    return nbits;
}

//Returns the largest number less than n whose first r binary digits are the
inline uint max_first_bits(uint n, uint r, uint m) {
    uint mod = 2 << r;
    uint carry = (m <= (n-1)%mod) ? 0 : (2 << r); 
    return ((((n-1) >> (r+1))) << (r+1)) + m - carry;
}

// Returns the largest number less than n whose first r binary digits are the
inline uint predic(uint r, uint round) {
    uint rounder = 1 << round;
    return (r % rounder) ^ (rounder>>1);
}

// Updates the current values of the characteristic coefficient given 
// the previous round of values.
inline void update(double x, double* curr, double* prev, uint k) {
    for (uint l = 1; l < k; l++) {
        curr[l] = x * prev[l-1] + prev[l];
    }
}

// The smaller of the two successors of a modulus.
uint min_succ(uint r, uint round) {
    uint mod = 1 << round;
    return r ^ mod;
}

// The larger of the two successors of a modulus.
uint max_succ(uint r, uint round) {
    uint mod = 1 << round;
    return (r ^ mod) + (mod << 1);
}

// Whether the integer x is a penultimate node in the computation tree.
bool is_penultimate(uint x, uint n, uint round) {
    uint mod = 2 << round;
    uint r = x % mod;
    return (x+mod >= n) && (min_succ(r, round) < n)
                        && (max_succ(r, round) >= n);
}

// Updates the gradient vector depending on the values of various parts of the
// tree.
void update_grad(VectorXd* grad, uint max, uint r, uint n, uint round, double val) {
    if (is_penultimate(max, n, round)) {
        (*grad)(min_succ(r, round)) = val;
        uint other_succ = max_succ(r, round);
        if (other_succ < n) {
            (*grad)(other_succ) = val;
        }
    }
}

/** 
 * Computes the gradient of the degree k elementary symmetric polynomial at x
 * using dynamic programming.
 *
 * We describe the method for this computation here.
 * D_i e_n^k(x) = e_{n-1}^{k-1}(x_^i), where x_^i is the vector obtained by
 * removing the i^th coordinate of x. We want to compute these usinr the 
 * recurrence
 * e_n^k(x) = x_i e_{n-1}^{k-1}(x_^i) + e_{n-1}^{k}(x_^i).
 *
 * If x_S is the subvector of x indexed by a set S, then we can compute
 * e_n^i(x_S) quickly if we know e_n^i(x_{S-i}) for i in [k].
 *
 * The algorithm divides the work into log(n) rounds. Each round computes
 * e_n^k(x_S) for some sets S, using the values computed in the previous round.
 *
 * In the first round,  e_n^i(x_S) is computed for i in [k], and
 * where S are the odd numbers less than n and the even numbers less than n.
 * odd integers.
 *
 */
VectorXd gradient(VectorXd x, uint k) {
    size_t n = x.rows();
    VectorXd grad(n);

    if (k == 0) {
        for (uint i = 0; i < n; i++) {
            grad(i) = 0;
        }
        return grad;
    }

    // Arrays storing the current and previous iterations of this method.
    double** prev = new double*[n]; 
    double** curr = new double*[n];
    for (uint i = 0; i < n; i++) {
        prev[i] = new double[k];
        curr[i] = new double[k];
    }

    // Initialize arrays
    for (uint i = 0; i < n; i++) {
        for (uint j = 1; j < k; j++) {
            prev[i][j] = 0;
        }
        prev[i][0] = 1;
        curr[i][0] = 1;
    }

    // Dynamic Programming Step
    for (uint round = 0; round < nbits(n); round++) {
        uint mod = 2 << round;
        uint limit = (mod > n) ? n : mod;
        // Each modulus is a separate computation.
        for (uint r = 0; r < limit; r++) {
            // Initialize first column of array, which is curr[r][:]
            // The previous modulus that should be used as the base case.
            uint prev_r = predic(r, round);
            uint old_max = (round == 0) ? r : max_first_bits(n,round-1,prev_r);
            // Initialize the first column of the array for this part.
            update(x(r), curr[r], prev[old_max], k);
            // Recurrence for the rest of the array.
            uint previous = r;
            for (uint i = r + mod; i < n; i += mod) {
                update(x(i), curr[i], curr[previous], k);
                previous = i;
            }
            update_grad(&grad, previous, r, n, round, curr[previous][k-1]);
        }
        // Swap the previous and current arrays.
        double** temp = prev;
        prev = curr;
        curr = temp;
    }

    // Clean up memory
    for (uint i = 0; i < n; i++) {
        delete[] prev[i];
        delete[] curr[i];
    }
    delete[] prev;
    delete[] curr;

    return grad;
}

VectorXd conditionals_eigen(VectorXd diagonal, VectorXd eigenvalues, MatrixXd eigenvectors, uint k) {
    // First, compute the gradient for the eigenvalues.
    VectorXd grad = gradient(eigenvalues, k-1);
    // Formula for the updates using the gradient
    VectorXd updates =
        (eigenvectors.array().square().matrix() *
        eigenvalues.asDiagonal() *
        eigenvalues.asDiagonal()) *
        grad;
    size_t n = eigenvalues.rows();
    VectorXd output(n);
    // Starting value.
    double val = esp(eigenvalues, k-1);
    // Then, we compute the results of this map for the Schur complements.
    return val * diagonal - updates;
}

VectorXd conditionals_inc(MatrixXd X, SelfAdjointEigenSolver<MatrixXd>* es, uint k) {
    es->compute(X);
    return conditionals_eigen(X.diagonal(), es->eigenvalues(), es->eigenvectors(), k);
}

VectorXd conditionals(MatrixXd X, uint k) {
    SelfAdjointEigenSolver<MatrixXd> es;
    return conditionals_inc(X, &es, k);
}
}
