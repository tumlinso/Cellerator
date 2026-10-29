#pragma once

#include <Cellerator/compute/tokenization/tokenization.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cellerator::tokenization {

enum class GeneAnnotationFormat : std::uint8_t { gtf, gff3 };

struct GeneRecord {
    std::string gene_id;
    std::string symbol;
    std::vector<std::string> aliases;
    std::optional<std::int64_t> token_id;
};

struct GeneVocabularyMetadata {
    std::string source;
    std::string organism;
    std::string assembly;
    std::string release;
};

struct GeneAnnotationOptions {
    std::string feature_type = "gene";
    GeneVocabularyMetadata metadata;
    unknown_identity_policy unknown_policy = unknown_identity_policy::error;
    std::optional<std::int64_t> unknown_token_id;
};

// Immutable string-to-integer gene identity table around the generic ATen
// vocabulary. String lookup is host-side; numerical identity mapping stays in
// ATen and retains its existing device and unknown-identity contracts.
class GeneVocabulary {
public:
    static GeneVocabulary from_records(
        std::vector<GeneRecord> records,
        GeneVocabularyMetadata metadata = {},
        unknown_identity_policy unknown_policy = unknown_identity_policy::error,
        std::optional<std::int64_t> unknown_token_id = {});

    static GeneVocabulary from_annotation(
        const std::string& path, GeneAnnotationFormat format,
        const GeneAnnotationOptions& options = {});

    static GeneVocabulary load(const std::string& path);
    void save(const std::string& path) const;

    std::int64_t resolve_identity(std::string_view gene_id_or_alias) const;
    std::vector<std::int64_t> resolve_identities(
        const std::vector<std::string>& gene_ids_or_aliases) const;

    const std::vector<GeneRecord>& genes() const noexcept { return genes_; }
    const GeneVocabularyMetadata& metadata() const noexcept { return metadata_; }

    // Return independent tensors so callers cannot mutate this vocabulary's
    // canonical identities or token assignment through ATen's shared storage.
    at::Tensor identity_tensor(at::Device device = at::Device(at::kCPU)) const;
    Vocabulary integer_vocabulary(
        at::Device device = at::Device(at::kCPU)) const;

private:
    GeneVocabulary(std::vector<GeneRecord> genes,
                   GeneVocabularyMetadata metadata,
                   unknown_identity_policy unknown_policy,
                   std::optional<std::int64_t> unknown_token_id);

    std::vector<GeneRecord> genes_;
    GeneVocabularyMetadata metadata_;
    unknown_identity_policy unknown_policy_;
    std::optional<std::int64_t> unknown_token_id_;
    std::unordered_map<std::string, std::int64_t> identities_by_id_;
    std::unordered_map<std::string, std::vector<std::int64_t>> identities_by_alias_;
};

} // namespace cellerator::tokenization
