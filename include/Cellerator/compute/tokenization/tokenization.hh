#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace cellerator::tokenization {

enum class unknown_identity_policy : std::uint8_t { error, drop, map_to_token };

struct Vocabulary {
    at::Tensor identities; // sorted, unique int64 caller identities
    at::Tensor token_ids; // int64 token IDs aligned to identities
    unknown_identity_policy unknown_policy = unknown_identity_policy::error;
    std::optional<std::int64_t> unknown_token_id;

    Vocabulary to(at::Device device) const;
};

struct TokenBatch {
    at::Tensor token_ids; // flat int64 [N]
    at::Tensor offsets; // int64 [sequence_count + 1]
    std::unordered_map<std::string, at::Tensor> attributes; // [N, ...]
};

struct PaddedBatch {
    at::Tensor token_ids; // int64 [B, max_length]
    at::Tensor lengths; // int64 [B]
    at::Tensor valid; // bool [B, max_length]
    std::unordered_map<std::string, at::Tensor> attributes; // [B, max_length, ...]
};

struct RankedFeatureOptions {
    std::int64_t max_per_row = -1;
    std::optional<at::Tensor> eligibility;
    std::optional<at::Tensor> scores; // dense [B,F] or CSR, aligned to values
    bool retain_measurements = false;
    bool retain_scores = false;
};

Vocabulary make_vocabulary(
    const at::Tensor& identities, const at::Tensor& token_ids,
    unknown_identity_policy policy = unknown_identity_policy::error,
    std::optional<std::int64_t> unknown_token_id = {});

TokenBatch map_identities(
    const at::Tensor& identity_ids, const at::Tensor& offsets,
    const Vocabulary& vocabulary,
    const std::unordered_map<std::string, at::Tensor>& attributes = {});

// Encode supplied events, preserving their order and repetitions.
TokenBatch event_stream(
    const at::Tensor& identity_ids, const at::Tensor& offsets,
    const Vocabulary& vocabulary,
    const std::unordered_map<std::string, at::Tensor>& attributes = {});

TokenBatch add_boundaries(
    const TokenBatch& batch,
    const std::optional<TokenBatch>& prefix = {},
    const std::optional<TokenBatch>& suffix = {});

// Dense input is [B,F]; CSR input has B rows. Feature identities are unique
// int64[F]. Explicit scores override values for ranking; eligibility defaults
// to original value != 0.
TokenBatch ranked_feature(
    const at::Tensor& values, const at::Tensor& feature_ids,
    const Vocabulary& vocabulary, const RankedFeatureOptions& options = {},
    const std::unordered_map<std::string, at::Tensor>& attributes = {});

// Normalizes with caller-supplied cell totals and fixed feature medians.
TokenBatch geneformer_style(
    const at::Tensor& counts, const at::Tensor& total_counts,
    const at::Tensor& feature_medians, const at::Tensor& feature_ids,
    const Vocabulary& vocabulary, std::int64_t max_per_row,
    const std::unordered_map<std::string, at::Tensor>& attributes = {},
    std::optional<at::ScalarType> compute_type = {});

PaddedBatch pad_sequences(const TokenBatch& batch, std::int64_t pad_token_id);
TokenBatch pack_padded(
    const at::Tensor& token_ids, const at::Tensor& lengths,
    const std::unordered_map<std::string, at::Tensor>& padded_attributes = {});

void validate_token_batch(const TokenBatch& batch);

} // namespace cellerator::tokenization
