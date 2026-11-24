#include <Cellerator/compute/tokenization/gene_tokenizer.hh>

#include <stdexcept>
#include <utility>

namespace cellerator::tokenization {

GeneTokenizer::~GeneTokenizer() = default;

GeneTokenizer::GeneTokenizer(std::shared_ptr<const GeneVocabulary> vocabulary)
    : vocabulary_(std::move(vocabulary)) {
    if (!vocabulary_) throw std::invalid_argument("gene tokenizer requires a vocabulary");
}

TokenBatch EventStreamGeneTokenizer::tokenize(
    const at::Tensor& gene_identity_ids, const at::Tensor& offsets,
    const std::unordered_map<std::string, at::Tensor>& attributes) const {
    return event_stream(gene_identity_ids, offsets,
                        vocabulary_for(gene_identity_ids.device()), attributes);
}

TokenBatch EventStreamGeneTokenizer::tokenize_names(
    const std::vector<std::string>& names, const at::Tensor& offsets,
    const std::unordered_map<std::string, at::Tensor>& attributes,
    at::Device device) const {
    auto identities = at::tensor(gene_vocabulary().resolve_identities(names),
                                 at::TensorOptions().dtype(at::kLong)).to(device);
    return tokenize(identities, offsets, attributes);
}

RankedGeneTokenizer::RankedGeneTokenizer(
    std::shared_ptr<const GeneVocabulary> vocabulary, RankedFeatureOptions options)
    : GeneTokenizer(std::move(vocabulary)), options_(std::move(options)) {
    if (options_.scores) options_.scores = options_.scores->clone();
    if (options_.eligibility) options_.eligibility = options_.eligibility->clone();
}

RankedFeatureOptions RankedGeneTokenizer::options() const {
    auto result = options_;
    if (result.scores) result.scores = result.scores->clone();
    if (result.eligibility) result.eligibility = result.eligibility->clone();
    return result;
}

TokenBatch RankedGeneTokenizer::tokenize(
    const at::Tensor& values, const at::Tensor& gene_identity_ids,
    const std::unordered_map<std::string, at::Tensor>& attributes) const {
    return ranked_feature(values, gene_identity_ids,
                          vocabulary_for(values.device()), options_, attributes);
}

TokenBatch RankedGeneTokenizer::geneformer_style(
    const at::Tensor& counts, const at::Tensor& total_counts,
    const at::Tensor& gene_medians, const at::Tensor& gene_identity_ids,
    const std::unordered_map<std::string, at::Tensor>& attributes,
    std::optional<at::ScalarType> compute_type) const {
    return tokenization::geneformer_style(
        counts, total_counts, gene_medians, gene_identity_ids,
        vocabulary_for(counts.device()), options_.max_per_row, attributes,
        compute_type);
}

} // namespace cellerator::tokenization
