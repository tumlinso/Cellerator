#include <Cellerator/interop/ggml/quantization.hh>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <string>
#include <vector>

namespace py = pybind11;
namespace ggml = cellerator::interop::ggml;

PYBIND11_MODULE(_ggml_native, module) {
    module.doc() = "Optional ggml CPU block-quantization bridge for Cellerator";

    py::class_<ggml::quantized_values>(module, "QuantizedValues")
        .def_property_readonly("format", [](const ggml::quantized_values& q) {
            return std::string(q.format);
        })
        .def_property_readonly("shape", [](const ggml::quantized_values& q) {
            py::tuple result(q.shape.size());
            for (std::size_t i = 0; i < q.shape.size(); ++i) {
                result[i] = q.shape[i];
            }
            return result;
        })
        .def_property_readonly("payload", [](const ggml::quantized_values& q) {
            return py::bytes(reinterpret_cast<const char*>(q.payload.data()), q.payload.size());
        })
        .def("dequantize", [](const ggml::quantized_values& q) {
            const auto values = ggml::dequantize(q);
            std::vector<py::ssize_t> shape;
            shape.reserve(q.shape.size());
            for (const auto extent : q.shape) {
                shape.push_back(static_cast<py::ssize_t>(extent));
            }
            py::array_t<float> result(shape);
            if (!values.empty()) {
                std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(float));
            }
            return result;
        });

    module.def("supported_formats", &ggml::supported_formats);
    module.def("quantize", [](py::array_t<float,
                               py::array::c_style | py::array::forcecast> input,
                               const std::string& format) {
        const auto info = input.request();
        std::vector<std::int64_t> shape;
        shape.reserve(static_cast<std::size_t>(info.ndim));
        for (py::ssize_t i = 0; i < info.ndim; ++i) {
            shape.push_back(static_cast<std::int64_t>(info.shape[static_cast<std::size_t>(i)]));
        }
        const auto* data = static_cast<const float*>(info.ptr);
        const auto count = static_cast<std::size_t>(info.size);
        return ggml::quantize(std::span<const float>(data, count), shape, format);
    }, py::arg("values"), py::arg("format"));
}
