#include "regression.h"
#include "conditionals.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

using sreg::regression;
using cond::conditionals;

PYBIND11_MODULE(regression, m) {
    m.def("regression", &regression, "A function that does sparse regression.");
    m.def("conditionals", &conditionals, "A function that computes conditionals of a given matrix.");
}
