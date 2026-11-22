#include <Cellerator/compute/tokenization/gene_tokenizer.hh>

#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace ce = cellerator;

namespace {
std::optional<at::ScalarType> compute_type_from_name(
    const std::optional<std::string>& name) {
    if (!name) return std::nullopt;
    if (*name == "float32") return at::kFloat;
    if (*name == "float64") return at::kDouble;
    throw std::invalid_argument("compute_type must be None, 'float32', or 'float64'");
}
} // namespace

void register_gene_tokenization(py::module_& module) {
    py::enum_<ce::tokenization::GeneAnnotationFormat>(module, "GeneAnnotationFormat")
        .value("gtf", ce::tokenization::GeneAnnotationFormat::gtf)
        .value("gff3", ce::tokenization::GeneAnnotationFormat::gff3);

    py::class_<ce::tokenization::GeneRecord>(module, "GeneRecord")
        .def(py::init<>())
        .def(py::init([](std::string gene_id, std::string symbol,
                         std::vector<std::string> aliases,
                         std::optional<std::int64_t> token_id) {
                return ce::tokenization::GeneRecord{
                    std::move(gene_id), std::move(symbol), std::move(aliases), token_id};
            }),
            py::arg("gene_id"), py::arg("symbol") = "",
            py::arg("aliases") = std::vector<std::string>{},
            py::arg("token_id") = std::nullopt)
        .def_readwrite("gene_id", &ce::tokenization::GeneRecord::gene_id)
        .def_readwrite("symbol", &ce::tokenization::GeneRecord::symbol)
        .def_readwrite("aliases", &ce::tokenization::GeneRecord::aliases)
        .def_readwrite("token_id", &ce::tokenization::GeneRecord::token_id);

    py::class_<ce::tokenization::GeneVocabularyMetadata>(module, "GeneVocabularyMetadata")
        .def(py::init<>())
        .def_readwrite("source", &ce::tokenization::GeneVocabularyMetadata::source)
        .def_readwrite("organism", &ce::tokenization::GeneVocabularyMetadata::organism)
        .def_readwrite("assembly", &ce::tokenization::GeneVocabularyMetadata::assembly)
        .def_readwrite("release", &ce::tokenization::GeneVocabularyMetadata::release);

    py::class_<ce::tokenization::GeneAnnotationOptions>(module, "GeneAnnotationOptions")
        .def(py::init<>())
        .def_readwrite("feature_type", &ce::tokenization::GeneAnnotationOptions::feature_type)
        .def_readwrite("metadata", &ce::tokenization::GeneAnnotationOptions::metadata)
        .def_readwrite("unknown_policy", &ce::tokenization::GeneAnnotationOptions::unknown_policy)
        .def_readwrite("unknown_token_id", &ce::tokenization::GeneAnnotationOptions::unknown_token_id);


    py::class_<ce::tokenization::GeneVocabulary,
               std::shared_ptr<ce::tokenization::GeneVocabulary>>(
        module, "GeneVocabulary")
        .def_static("from_records", &ce::tokenization::GeneVocabulary::from_records,
            py::arg("records"), py::arg("metadata") = ce::tokenization::GeneVocabularyMetadata{},
            py::arg("unknown_policy") = ce::tokenization::unknown_identity_policy::error,
            py::arg("unknown_token_id") = std::nullopt)
        .def_static("from_annotation", &ce::tokenization::GeneVocabulary::from_annotation,
            py::arg("path"), py::arg("format"),
            py::arg("options") = ce::tokenization::GeneAnnotationOptions{})
        .def_static("load", &ce::tokenization::GeneVocabulary::load, py::arg("path"))
        .def("save", &ce::tokenization::GeneVocabulary::save, py::arg("path"))
        .def("resolve_identity", &ce::tokenization::GeneVocabulary::resolve_identity,
             py::arg("gene_id_or_alias"))
        .def("resolve_identities", &ce::tokenization::GeneVocabulary::resolve_identities,
             py::arg("gene_ids_or_aliases"))
        .def("identity_tensor", [](const ce::tokenization::GeneVocabulary& vocabulary,
                                    const std::string& device) {
            return vocabulary.identity_tensor(at::Device(device));
        }, py::arg("device") = "cpu")
        .def("integer_vocabulary", [](const ce::tokenization::GeneVocabulary& vocabulary,
                                      const std::string& device) {
            return vocabulary.integer_vocabulary(at::Device(device));
        }, py::arg("device") = "cpu")
        .def_property_readonly("genes", [](const ce::tokenization::GeneVocabulary& value) {
            return std::vector<ce::tokenization::GeneRecord>(value.genes());
        })
        .def_property_readonly("metadata", [](const ce::tokenization::GeneVocabulary& value) {
            return ce::tokenization::GeneVocabularyMetadata(value.metadata());
        });


    py::class_<ce::tokenization::GeneTokenizer,
               std::shared_ptr<ce::tokenization::GeneTokenizer>>(
        module, "GeneTokenizer")
        .def_property_readonly("gene_vocabulary",
            &ce::tokenization::GeneTokenizer::gene_vocabulary,
            py::return_value_policy::reference_internal);

    py::class_<ce::tokenization::EventStreamGeneTokenizer,
               ce::tokenization::GeneTokenizer,
               std::shared_ptr<ce::tokenization::EventStreamGeneTokenizer>>(
        module, "EventStreamGeneTokenizer")
        .def(py::init([](std::shared_ptr<ce::tokenization::GeneVocabulary> vocabulary) {
            return std::make_shared<ce::tokenization::EventStreamGeneTokenizer>(
                std::move(vocabulary));
        }), py::arg("vocabulary"))
        .def("tokenize", &ce::tokenization::EventStreamGeneTokenizer::tokenize,
            py::arg("gene_identity_ids"), py::arg("offsets"),
            py::arg("attributes") =
                std::unordered_map<std::string, at::Tensor>{})
        .def("tokenize_names", [](const ce::tokenization::EventStreamGeneTokenizer& tokenizer,
                                  const std::vector<std::string>& names,
                                  const at::Tensor& offsets,
                                  const std::unordered_map<std::string, at::Tensor>& attributes,
                                  const std::string& device) {
            return tokenizer.tokenize_names(names, offsets, attributes, at::Device(device));
        }, py::arg("gene_ids_or_aliases"), py::arg("offsets"),
           py::arg("attributes") = std::unordered_map<std::string, at::Tensor>{},
           py::arg("device") = "cpu");

    py::class_<ce::tokenization::RankedGeneTokenizer,
               ce::tokenization::GeneTokenizer,
               std::shared_ptr<ce::tokenization::RankedGeneTokenizer>>(
        module, "RankedGeneTokenizer")
        .def(py::init([](std::shared_ptr<ce::tokenization::GeneVocabulary> vocabulary,
                         const ce::tokenization::RankedFeatureOptions& options) {
            return std::make_shared<ce::tokenization::RankedGeneTokenizer>(
                std::move(vocabulary), options);
        }), py::arg("vocabulary"),
           py::arg("options") = ce::tokenization::RankedFeatureOptions{})
        .def_property_readonly("options", &ce::tokenization::RankedGeneTokenizer::options)
        .def("tokenize", &ce::tokenization::RankedGeneTokenizer::tokenize,
            py::arg("values"), py::arg("gene_identity_ids"),
            py::arg("attributes") =
                std::unordered_map<std::string, at::Tensor>{})
        .def("geneformer_style", [](const ce::tokenization::RankedGeneTokenizer& tokenizer,
                                    const at::Tensor& counts,
                                    const at::Tensor& total_counts,
                                    const at::Tensor& gene_medians,
                                    const at::Tensor& gene_identity_ids,
                                    const std::unordered_map<std::string, at::Tensor>& attributes,
                                    const std::optional<std::string>& compute_type) {
            return tokenizer.geneformer_style(counts, total_counts, gene_medians,
                gene_identity_ids, attributes, compute_type_from_name(compute_type));
        }, py::arg("counts"), py::arg("total_counts"), py::arg("gene_medians"),
           py::arg("gene_identity_ids"),
           py::arg("attributes") = std::unordered_map<std::string, at::Tensor>{},
           py::arg("compute_type") = std::nullopt);
}
