#include <Cellerator/compute/operators/aten_ops.hh>
#include <Cellerator/compute/tokenization/sequence_buffer.hh>
#include <Cellerator/compute/tokenization/tokenization.hh>

#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace ce = cellerator;

void register_gene_tokenization(pybind11::module_& module);

namespace {
std::optional<at::ScalarType> compute_type_from_name(
    const std::optional<std::string>& name) {
    if (!name) return std::nullopt;
    if (*name == "float32") return at::kFloat;
    if (*name == "float64") return at::kDouble;
    throw std::invalid_argument("compute_type must be None, 'float32', or 'float64'");
}
} // namespace

PYBIND11_MODULE(_tokenization, module) {
    module.doc() = "ATen backed Cellerator tokenization primitives";

    py::enum_<ce::tokenization::unknown_identity_policy>(
        module, "UnknownIdentityPolicy")
        .value("error", ce::tokenization::unknown_identity_policy::error)
        .value("drop", ce::tokenization::unknown_identity_policy::drop)
        .value("map_to_token",
            ce::tokenization::unknown_identity_policy::map_to_token);

    py::class_<ce::tokenization::Vocabulary>(module, "Vocabulary")
        .def("to", [](const ce::tokenization::Vocabulary& vocabulary,
                       const std::string& device) {
            return vocabulary.to(at::Device(device));
        }, py::arg("device"))
        .def_readonly("identities", &ce::tokenization::Vocabulary::identities)
        .def_readonly("token_ids", &ce::tokenization::Vocabulary::token_ids)
        .def_readonly("unknown_policy",
            &ce::tokenization::Vocabulary::unknown_policy)
        .def_readonly("unknown_token_id",
            &ce::tokenization::Vocabulary::unknown_token_id);

    py::class_<ce::tokenization::RankedFeatureOptions>(
        module, "RankedFeatureOptions")
        .def(py::init<>())
        .def_readwrite("max_per_row",
            &ce::tokenization::RankedFeatureOptions::max_per_row)
        .def_readwrite("eligibility",
            &ce::tokenization::RankedFeatureOptions::eligibility)
        .def_readwrite("scores",
            &ce::tokenization::RankedFeatureOptions::scores)
        .def_readwrite("retain_measurements",
            &ce::tokenization::RankedFeatureOptions::retain_measurements)
        .def_readwrite("retain_scores",
            &ce::tokenization::RankedFeatureOptions::retain_scores);

    register_gene_tokenization(module);

    py::class_<ce::tokenization::TokenBatch>(module, "TokenBatch")
        .def(py::init([](const at::Tensor& token_ids,
                         const at::Tensor& offsets,
                         const std::unordered_map<std::string, at::Tensor>& attributes) {
                return ce::tokenization::TokenBatch{
                    token_ids, offsets, attributes};
            }),
            py::arg("token_ids"), py::arg("offsets"),
            py::arg("attributes") =
                std::unordered_map<std::string, at::Tensor>{})
        .def_readonly("token_ids", &ce::tokenization::TokenBatch::token_ids)
        .def_readonly("offsets", &ce::tokenization::TokenBatch::offsets)
        .def_readonly("attributes", &ce::tokenization::TokenBatch::attributes);
    py::class_<ce::aten_ops::RankedSelection>(module, "RankedSelection")
        .def_readonly("flat_indices", &ce::aten_ops::RankedSelection::flat_indices)
        .def_readonly("offsets", &ce::aten_ops::RankedSelection::offsets);
    py::class_<ce::aten_ops::FeatureStatistics>(module, "FeatureStatistics")
        .def_readonly("medians", &ce::aten_ops::FeatureStatistics::medians)
        .def_readonly("nonzero_counts",
            &ce::aten_ops::FeatureStatistics::nonzero_counts)
        .def_readonly("valid", &ce::aten_ops::FeatureStatistics::valid);

    py::class_<ce::tokenization::PaddedBatch>(module, "PaddedBatch")
        .def_readonly("token_ids", &ce::tokenization::PaddedBatch::token_ids)
        .def_readonly("lengths", &ce::tokenization::PaddedBatch::lengths)
        .def_readonly("valid", &ce::tokenization::PaddedBatch::valid)
        .def_readonly("attributes", &ce::tokenization::PaddedBatch::attributes);

    py::class_<ce::tokenization::AppendResult>(module, "AppendResult")
        .def_readonly("succeeded", &ce::tokenization::AppendResult::succeeded)
        .def_readonly("lengths", &ce::tokenization::AppendResult::lengths);

    py::class_<ce::tokenization::SequenceBuffer>(module, "SequenceBuffer")
        .def_static("create",
            [](const std::vector<std::int64_t>& capacities,
                const std::unordered_map<std::string, at::Tensor>& examples,
                const std::string& device) {
                return ce::tokenization::SequenceBuffer::create(
                    capacities, examples, at::Device(device));
            },
            py::arg("capacities"), py::arg("attribute_examples") =
                std::unordered_map<std::string, at::Tensor>{},
            py::arg("device") = "cpu")
        .def_property_readonly("sequence_count",
            &ce::tokenization::SequenceBuffer::sequence_count)
        .def_property_readonly("lengths",
            &ce::tokenization::SequenceBuffer::lengths)
        .def_property_readonly("token_storage",
            &ce::tokenization::SequenceBuffer::token_storage)
        .def_property_readonly("attribute_storage",
            &ce::tokenization::SequenceBuffer::attribute_storage)
        .def("append_into", &ce::tokenization::SequenceBuffer::append_into,
            py::arg("selected_rows"), py::arg("updates"),
            py::arg("status_buffer") = std::nullopt)
        .def("append_checked",
            &ce::tokenization::SequenceBuffer::append_checked,
            py::arg("selected_rows"), py::arg("updates"))
        .def("reserve", &ce::tokenization::SequenceBuffer::reserve,
            py::arg("new_capacities"))
        .def("compact", &ce::tokenization::SequenceBuffer::compact);

    module.def("make_vocabulary", &ce::tokenization::make_vocabulary,
        py::arg("identities"), py::arg("token_ids"),
        py::arg("unknown_policy") =
            ce::tokenization::unknown_identity_policy::error,
        py::arg("unknown_token_id") = std::nullopt);
    module.def("map_identities", &ce::tokenization::map_identities,
        py::arg("identity_ids"), py::arg("offsets"), py::arg("vocabulary"),
        py::arg("attributes") =
            std::unordered_map<std::string, at::Tensor>{});
    module.def("event_stream", &ce::tokenization::event_stream,
        py::arg("identity_ids"), py::arg("offsets"), py::arg("vocabulary"),
        py::arg("attributes") =
            std::unordered_map<std::string, at::Tensor>{});
    module.def("event_stream_into", &ce::tokenization::event_stream_into,
        py::arg("output"), py::arg("selected_rows"),
        py::arg("identity_ids"), py::arg("offsets"),
        py::arg("vocabulary"),
        py::arg("attributes") =
            std::unordered_map<std::string, at::Tensor>{},
        py::arg("status_buffer") = std::nullopt);
    module.def("ranked_feature", &ce::tokenization::ranked_feature,
        py::arg("values"), py::arg("feature_ids"), py::arg("vocabulary"),
        py::arg("options") = ce::tokenization::RankedFeatureOptions{},
        py::arg("attributes") =
            std::unordered_map<std::string, at::Tensor>{});
    module.def("ranked_feature_into", &ce::tokenization::ranked_feature_into,
        py::arg("output"), py::arg("selected_rows"), py::arg("values"),
        py::arg("feature_ids"), py::arg("vocabulary"),
        py::arg("options") = ce::tokenization::RankedFeatureOptions{},
        py::arg("attributes") =
            std::unordered_map<std::string, at::Tensor>{},
        py::arg("status_buffer") = std::nullopt);
    module.def("geneformer_style",
        [](const at::Tensor& counts, const at::Tensor& total_counts,
            const at::Tensor& feature_medians, const at::Tensor& feature_ids,
            const ce::tokenization::Vocabulary& vocabulary,
            std::int64_t max_per_row,
            const std::unordered_map<std::string, at::Tensor>& attributes,
            const std::optional<std::string>& compute_type) {
            return ce::tokenization::geneformer_style(
                counts, total_counts, feature_medians, feature_ids, vocabulary,
                max_per_row, attributes, compute_type_from_name(compute_type));
        },
        py::arg("counts"), py::arg("total_counts"),
        py::arg("feature_medians"), py::arg("feature_ids"),
        py::arg("vocabulary"), py::arg("max_per_row"),
        py::arg("attributes") =
            std::unordered_map<std::string, at::Tensor>{},
        py::arg("compute_type") = std::nullopt);
    module.def("pad_sequences", &ce::tokenization::pad_sequences,
        py::arg("batch"), py::arg("pad_token_id"));
    module.def("pack_padded", &ce::tokenization::pack_padded,
        py::arg("token_ids"), py::arg("lengths"),
        py::arg("padded_attributes") =
            std::unordered_map<std::string, at::Tensor>{});
    module.def("add_boundaries", &ce::tokenization::add_boundaries,
        py::arg("batch"), py::arg("prefix") = std::nullopt,
        py::arg("suffix") = std::nullopt);
    module.def("normalize_features",
        [](const at::Tensor& values,
            std::optional<at::Tensor> row_totals,
            std::optional<at::Tensor> feature_factors,
            double target_sum,
            const std::optional<std::string>& compute_type) {
            return ce::aten_ops::normalize_features(values, row_totals,
                feature_factors, target_sum,
                compute_type_from_name(compute_type));
        }, py::arg("values"), py::arg("row_totals") = std::nullopt,
        py::arg("feature_factors") = std::nullopt,
        py::arg("target_sum") = 1.0, py::arg("compute_type") = std::nullopt);
    module.def("select_ranked_features",
        &ce::aten_ops::select_ranked_features,
        py::arg("scores"), py::arg("feature_ids"),
        py::arg("eligibility") = std::nullopt,
        py::arg("max_per_row") = -1);
    module.def("nonzero_feature_median",
        [](const std::vector<at::Tensor>& batches, std::int64_t block_size,
            const std::optional<std::string>& compute_type) {
            return ce::aten_ops::nonzero_feature_median(batches, block_size,
                compute_type_from_name(compute_type));
        }, py::arg("batches"), py::arg("feature_block_size") = 256,
        py::arg("compute_type") = std::nullopt);
}
