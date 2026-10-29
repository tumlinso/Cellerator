#include <Cellerator/bindings/torch/mechanism.hh>

#ifdef CELLERATOR_HAS_SPARSE_SUPPORT
#include "../../sparse_support/module.hh"
#endif

#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace ce = cellerator::bindings::torch;

PYBIND11_MODULE(_torch, m) {
    m.doc() = "Optional PyTorch adapters for Cellerator native owners";
    m.def("coefficients", &ce::coefficients, py::arg("handle"));
    m.def("validate_coefficients", &ce::validate_coefficients,
          py::arg("coefficients"), py::arg("handle"));
    m.def("mechanism_apply", &ce::mechanism_apply,
          py::arg("input"), py::arg("coefficients"), py::arg("handle"), py::arg("axis_words"));
    m.def("is_native_coefficient", &ce::is_native_coefficient,
          py::arg("tensor"), py::arg("handles"));
    m.def("begin_update", &ce::begin_update, py::arg("handle"));
    m.def("publish_update", &ce::publish_update, py::arg("handle"));
    m.def("synchronize_coefficients", &ce::synchronize_coefficients, py::arg("handle"));
#ifdef CELLERATOR_HAS_SPARSE_SUPPORT
    bind_sparse_support(m);
    m.attr("has_sparse_support") = true;
#else
    m.attr("has_sparse_support") = false;
#endif
}
