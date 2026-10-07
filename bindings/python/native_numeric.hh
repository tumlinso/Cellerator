#pragma once

#include <pybind11/pybind11.h>

namespace cellerator::bindings::python {
void bind_resident_cuda(pybind11::module_& module);
}
