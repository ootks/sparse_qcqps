#include "regression.h"
#include "pca.h"
#include "conditionals.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

using sreg::regression;
using spca::pca;
using cond::conditionals;

PYBIND11_MODULE(lpm_methods, m) {
    m.def("pca", &pca, "A function that does sparse pca.");
    m.def("regression", &regression, "A function that does sparse regression.");
    m.def("conditionals", &conditionals, "A function that computes conditionals of a given matrix.");
}
