#include <Cellerator/nn/attention.hh>
#include <Cellerator/nn/features.hh>
#include <Cellerator/nn/manifest.hh>
#include <Cellerator/nn/ops.hh>

#include <torch/extension.h>
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace nn = cellerator::nn;

namespace {

at::Tensor optional_tensor(const py::handle& value) {
    if (value.is_none()) return {};
    return py::cast<at::Tensor>(value);
}

std::optional<std::int64_t> optional_size(
    const py::handle& requested, const at::Tensor& projection) {
    if (!requested.is_none()) return py::cast<std::int64_t>(requested);
    if (projection.defined()) return projection.size(1);
    return std::nullopt;
}

nn::LinearParameters borrowed_linear_parameters(
    const py::sequence& parameters, py::ssize_t weight_index,
    py::ssize_t bias_index) {
    return nn::LinearParameters(
        py::cast<at::Tensor>(parameters[weight_index]),
        optional_tensor(parameters[bias_index]));
}

nn::AttentionParameters borrowed_attention_parameters(const py::sequence& values) {
    if (py::len(values) != 7) {
        throw py::value_error(
            "attention parameters must be [q_weight, q_bias, k_weight, k_bias, "
            "v_weight, v_bias, output_weight]");
    }
    nn::AttentionParameters parameters;
    parameters.query = borrowed_linear_parameters(values, 0, 1);
    parameters.key = borrowed_linear_parameters(values, 2, 3);
    parameters.value = borrowed_linear_parameters(values, 4, 5);
    parameters.output = nn::LinearParameters(
        py::cast<at::Tensor>(values[6]));
    return parameters;
}

} // namespace

PYBIND11_MODULE(_nn_native, module) {
    module.doc() = "Functional ATen entry points for Cellerator's native NN foundation";

    module.def("encode_features",
        [](const at::Tensor& ids,
           const py::object& values,
           const py::object& observed,
           const py::object& attributes,
           const at::Tensor& feature_embedding,
           const at::Tensor& status_embedding,
           const py::object& value_projection,
           const py::object& attribute_projection,
           const py::object& value_size,
           const py::object& attribute_size) {
            nn::FeatureEncoderParameters parameters;
            parameters.feature_embedding = feature_embedding;
            parameters.status_embedding = status_embedding;
            parameters.value_projection = optional_tensor(value_projection);
            parameters.attribute_projection = optional_tensor(attribute_projection);

            nn::FeatureEncoderOptions options;
            if (feature_embedding.dim() != 2) {
                throw py::value_error("feature_embedding must have shape [vocabulary, width]");
            }
            options.vocabulary_size = feature_embedding.size(0);
            options.width = feature_embedding.size(1);
            options.value_size = optional_size(value_size, parameters.value_projection);
            options.attribute_size = optional_size(
                attribute_size, parameters.attribute_projection);

            nn::FeatureInput input;
            input.ids = ids;
            if (!values.is_none()) input.values = py::cast<at::Tensor>(values);
            if (!observed.is_none()) input.observed = py::cast<at::Tensor>(observed);
            if (!attributes.is_none()) input.attributes = py::cast<at::Tensor>(attributes);
            return nn::encode_features(options, parameters, input);
        },
        py::arg("ids"),
        py::kw_only(),
        py::arg("values") = py::none(),
        py::arg("observed") = py::none(),
        py::arg("attributes") = py::none(),
        py::arg("feature_embedding"),
        py::arg("status_embedding"),
        py::arg("value_projection") = py::none(),
        py::arg("attribute_projection") = py::none(),
        py::arg("value_size") = py::none(),
        py::arg("attribute_size") = py::none());

    module.def("relation_attention",
        [](const at::Tensor& source,
           const at::Tensor& target,
           const py::sequence& parameter_values,
           std::int64_t width,
           std::int64_t heads,
           double dropout,
           const std::vector<std::int64_t>& source_offsets,
           const std::vector<std::int64_t>& target_offsets,
           bool causal,
           bool training) {
            auto parameters = borrowed_attention_parameters(parameter_values);
            nn::AttentionOptions options;
            options.width = width;
            options.heads = heads;
            options.dropout = dropout;
            options.causal = causal;

            nn::AttentionLayout layout;
            layout.source_offsets = source_offsets;
            layout.target_offsets = target_offsets;
            layout.causal = causal;
            return nn::relation_attention(source, target, parameters, options,
                layout, nn::Run{training});
        },
        py::arg("source"), py::arg("target"), py::arg("parameters"),
        py::arg("width"), py::arg("heads"), py::arg("dropout") = 0.0,
        py::arg("source_offsets") = std::vector<std::int64_t>{},
        py::arg("target_offsets") = std::vector<std::int64_t>{},
        py::arg("causal") = false, py::arg("training") = false);

    module.def("sparse_attention",
        [](const at::Tensor& query, const at::Tensor& key,
           const at::Tensor& value, const at::Tensor& rows,
           const at::Tensor& columns, double scale, double dropout,
           bool training) {
            return nn::sparse_attention(query, key, value, rows, columns,
                scale, dropout, nn::Run{training});
        },
        py::arg("query"), py::arg("key"), py::arg("value"),
        py::arg("rows"), py::arg("columns"), py::arg("scale"),
        py::arg("dropout") = 0.0, py::arg("training") = false);

    module.def("canonical_biological_manifest",
        [](const std::string& json_text) {
            return nn::serialize_biological_manifest(
                nn::parse_biological_manifest(json_text));
        },
        py::arg("json_text"));

    module.def("product2_cpu", &nn::ops::product2_cpu,
        py::arg("input"), py::arg("coefficient"), py::arg("lhs"), py::arg("rhs"));

#ifdef CELLERATOR_NN_HAS_CUDA
    module.def("grouped_linear_cuda", &nn::ops::grouped_linear_cuda,
        py::arg("input"), py::arg("weights"), py::arg("row_offsets"));
    module.def("csr_scaled_dot_product_attention_cuda",
        &nn::ops::csr_scaled_dot_product_attention_cuda,
        py::arg("query"), py::arg("key"), py::arg("value"),
        py::arg("offsets"), py::arg("key_ids"), py::arg("scale"),
        py::arg("dropout") = 0.0);
    module.def("csr_scaled_dot_product_attention_cuda_prepared",
        &nn::ops::csr_scaled_dot_product_attention_cuda_prepared,
        py::arg("query"), py::arg("key"), py::arg("value"),
        py::arg("offsets"), py::arg("key_ids"), py::arg("scale"),
        py::arg("dropout") = 0.0);
#endif
}
