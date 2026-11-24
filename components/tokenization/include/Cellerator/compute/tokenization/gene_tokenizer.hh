#pragma once

#include <Cellerator/compute/tokenization/gene_vocabulary.hh>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cellerator::tokenization {

// Shared ownership and vocabulary contract for gene tokenization. Concrete
// mechanisms deliberately keep their typed execution methods; a common virtual
// input signature remains open until their input contracts are unified.
class GeneTokenizer {
public:
    virtual ~GeneTokenizer() = 0;

    const GeneVocabulary& gene_vocabulary() const noexcept { return *vocabulary_; }

protected:
    explicit GeneTokenizer(std::shared_ptr<const GeneVocabulary> vocabulary);

    Vocabulary vocabulary_for(at::Device device) const {
        return vocabulary_->integer_vocabulary(device);
    }

private:
    std::shared_ptr<const GeneVocabulary> vocabulary_;
};

class EventStreamGeneTokenizer final : public GeneTokenizer {
public:
    explicit EventStreamGeneTokenizer(
        std::shared_ptr<const GeneVocabulary> vocabulary)
        : GeneTokenizer(std::move(vocabulary)) {}

    TokenBatch tokenize(
        const at::Tensor& gene_identity_ids, const at::Tensor& offsets,
        const std::unordered_map<std::string, at::Tensor>& attributes = {}) const;

    TokenBatch tokenize_names(
        const std::vector<std::string>& gene_ids_or_aliases,
        const at::Tensor& offsets,
        const std::unordered_map<std::string, at::Tensor>& attributes = {},
        at::Device device = at::Device(at::kCPU)) const;
};

class RankedGeneTokenizer final : public GeneTokenizer {
public:
    explicit RankedGeneTokenizer(
        std::shared_ptr<const GeneVocabulary> vocabulary,
        RankedFeatureOptions options = {});

    RankedFeatureOptions options() const;

    TokenBatch tokenize(
        const at::Tensor& values, const at::Tensor& gene_identity_ids,
        const std::unordered_map<std::string, at::Tensor>& attributes = {}) const;

    TokenBatch geneformer_style(
        const at::Tensor& counts, const at::Tensor& total_counts,
        const at::Tensor& gene_medians, const at::Tensor& gene_identity_ids,
        const std::unordered_map<std::string, at::Tensor>& attributes = {},
        std::optional<at::ScalarType> compute_type = {}) const;

private:
    RankedFeatureOptions options_;
};

} // namespace cellerator::tokenization
