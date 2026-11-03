#include <Cellerator/nn/attention.hh>
#include <Cellerator/nn/features.hh>
#include <Cellerator/nn/manifest.hh>
#include <Cellerator/nn/ops.hh>

#include <torch/extension.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef CELLERATOR_NN_HAS_CUDA
#include <Cellerator/memory/cuda/resources.hh>
#include <Cellerator/modeling/cuda/factorized_interaction.hh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <mutex>
#endif

namespace py = pybind11;
namespace nn = cellerator::nn;
namespace layout = cellerator::layout;
#ifdef CELLERATOR_NN_HAS_CUDA
namespace modeling = cellerator::modeling;
namespace memory = cellerator::memory;
#endif

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

#ifdef CELLERATOR_NN_HAS_CUDA
layout::FactorizedInteractionLayout parse_factorized_interaction_plan(
    const py::iterable& records,
    std::uint64_t inputs, std::uint64_t contexts, std::uint64_t outputs,
    std::uint64_t alpha_rows, std::uint64_t biases, std::uint64_t gammas,
    std::uint64_t pattern_id, std::uint64_t structure_id,
    std::uint64_t structure_epoch, std::uint32_t max_mechanisms,
    std::uint32_t max_inputs, std::uint32_t max_contexts) {
    std::vector<layout::FactorizedInteractionMechanism> mechanisms;
    for (const auto& item : records) {
        if (!py::isinstance<py::dict>(item))
            throw py::type_error("each interaction record must be a mapping");
        const auto record = py::reinterpret_borrow<py::dict>(item);
        auto required = [&](const char* key) -> py::handle {
            if (!record.contains(key))
                throw py::key_error(std::string("interaction record is missing '") + key + "'");
            return record[py::str(key)];
        };

        layout::FactorizedInteractionMechanism mechanism;
        mechanism.logical_id = py::cast<std::uint64_t>(required("logical_id"));
        const auto role_inputs = py::cast<std::vector<std::uint64_t>>(required("inputs"));
        if (role_inputs.size() < 2 ||
            role_inputs.size() > layout::factorized_interaction_max_arity)
            throw py::value_error("interaction inputs must contain 2, 3, or 4 ordered roles");
        mechanism.arity = static_cast<std::uint32_t>(role_inputs.size());
        std::copy(role_inputs.begin(), role_inputs.end(), mechanism.inputs.begin());
        mechanism.target = py::cast<std::uint64_t>(required("target"));
        mechanism.alpha = py::cast<std::uint64_t>(required("alpha"));
        mechanism.bias = py::cast<std::uint64_t>(required("bias"));
        if (record.contains("terms") && !record["terms"].is_none()) {
            for (const auto& term_value : py::reinterpret_borrow<py::iterable>(record["terms"])) {
                const auto term = py::cast<std::pair<std::uint64_t, std::uint64_t>>(term_value);
                mechanism.terms.push_back({term.first, term.second});
            }
        }
        mechanisms.push_back(std::move(mechanism));
    }

    layout::FactorizedInteractionExtents extents{
        inputs, contexts, outputs, alpha_rows, biases, gammas};
    layout::FactorizedInteractionOptions options;
    options.pattern_id = pattern_id;
    options.structure_id = structure_id;
    options.structure_epoch = structure_epoch;
    options.max_mechanisms = max_mechanisms;
    options.max_inputs = max_inputs;
    options.max_contexts = max_contexts;
    return layout::prepare_factorized_interactions(mechanisms, extents, options);
}

class PreparedFactorizedInteraction {
public:
    explicit PreparedFactorizedInteraction(layout::FactorizedInteractionLayout plan)
        : plan_(std::move(plan)) {}

    std::shared_ptr<modeling::cuda::FactorizedInteractionDeviceLayout> on_device(
        const at::Tensor& tensor) const {
        TORCH_CHECK(tensor.is_cuda(),
                    "prepared factorized interaction requires CUDA tensors");
        c10::cuda::CUDAGuard guard(tensor.device());
        const int device = tensor.get_device();
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = device_layouts_.find(device);
        if (found != device_layouts_.end()) return found->second;
        const auto torch_stream = c10::cuda::getCurrentCUDAStream(device);
        auto stream = memory::cuda::Stream::borrow(torch_stream.stream(), device);
        auto owner = std::make_shared<modeling::cuda::FactorizedInteractionDeviceLayout>(
            plan_, device, stream);
        device_layouts_.emplace(device, owner);
        return owner;
    }

private:
    layout::FactorizedInteractionLayout plan_;
    mutable std::mutex mutex_;
    mutable std::unordered_map<
        int, std::shared_ptr<modeling::cuda::FactorizedInteractionDeviceLayout>>
        device_layouts_;
};

at::ScalarType factorized_output_type(const std::string& dtype) {
    if (dtype == "float16") return at::kHalf;
    if (dtype == "float32") return at::kFloat;
    throw py::value_error("output_dtype must be 'float16' or 'float32'");
}

bool factorized_direct_route(const std::string& route) {
    if (route == "packet") return false;
    if (route == "direct") return true;
    throw py::value_error("route must be 'packet' or 'direct'");
}
#endif

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

    py::class_<PreparedFactorizedInteraction,
               std::shared_ptr<PreparedFactorizedInteraction>>(
        module, "PreparedFactorizedInteraction")
        .def(py::init([](
            const py::iterable& records,
            std::uint64_t inputs, std::uint64_t contexts, std::uint64_t outputs,
            std::uint64_t alpha_rows, std::uint64_t biases, std::uint64_t gammas,
            std::uint64_t pattern_id, std::uint64_t structure_id,
            std::uint64_t structure_epoch, std::uint32_t max_mechanisms,
            std::uint32_t max_inputs, std::uint32_t max_contexts) {
            auto plan = parse_factorized_interaction_plan(
                records, inputs, contexts, outputs, alpha_rows, biases, gammas,
                pattern_id, structure_id, structure_epoch, max_mechanisms,
                max_inputs, max_contexts);
            return std::make_shared<PreparedFactorizedInteraction>(std::move(plan));
        }),
            py::arg("records"), py::arg("inputs"), py::arg("contexts"),
            py::arg("outputs"), py::arg("alpha_rows"), py::arg("biases"),
            py::arg("gammas"), py::arg("pattern_id") = 1,
            py::arg("structure_id") = 1, py::arg("structure_epoch") = 1,
            py::arg("max_mechanisms") = layout::factorized_interaction_max_packet_mechanisms,
            py::arg("max_inputs") = layout::factorized_interaction_max_packet_inputs,
            py::arg("max_contexts") = layout::factorized_interaction_max_packet_contexts);

    module.def("factorized_interaction_cuda",
        [](const PreparedFactorizedInteraction& plan,
           const at::Tensor& x, const at::Tensor& z,
           const at::Tensor& alpha, const at::Tensor& bias,
           const at::Tensor& gamma, const std::string& output_dtype,
           const std::string& route) {
            const auto selected_dtype = factorized_output_type(output_dtype);
            const auto selected_route = factorized_direct_route(route);
            auto owner = plan.on_device(x);
            return nn::ops::factorized_interaction_cuda_prepared(
                owner, x, z, alpha, bias, gamma,
                selected_dtype, selected_route);
        },
        py::arg("plan"), py::arg("x"), py::arg("z"), py::arg("alpha"),
        py::arg("bias"), py::arg("gamma"), py::arg("output_dtype") = "float16",
        py::arg("route") = "packet");
#endif
}
