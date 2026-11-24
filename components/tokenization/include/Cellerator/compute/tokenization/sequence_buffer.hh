#pragma once

#include <Cellerator/compute/tokenization/tokenization.hh>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cellerator::tokenization {

struct AppendResult {
    at::Tensor succeeded; // bool [submitted sequences], on buffer device
    at::Tensor lengths; // view of current lengths
};

class SequenceBuffer {
public:
    static SequenceBuffer create(
        const std::vector<std::int64_t>& capacities,
        const std::unordered_map<std::string, at::Tensor>& attribute_examples = {},
        at::Device device = at::Device(at::kCPU));

    std::int64_t sequence_count() const noexcept;
    at::Tensor lengths() const;
    at::Tensor token_storage() const;
    const std::unordered_map<std::string, at::Tensor>& attribute_storage() const noexcept;

    AppendResult append_into(const at::Tensor& selected_rows, const TokenBatch& updates,
                             std::optional<at::Tensor> status_buffer = {});
    void append_checked(const at::Tensor& selected_rows, const TokenBatch& updates);
    SequenceBuffer reserve(const std::vector<std::int64_t>& new_capacities) const;
    TokenBatch compact() const;

private:
    std::vector<std::int64_t> capacities_host_;
    std::vector<std::int64_t> starts_host_;
    at::Tensor starts_;
    at::Tensor capacities_;
    at::Tensor row_owner_scratch_;
    at::Tensor append_error_scratch_;
    at::Tensor lengths_;
    at::Tensor token_storage_;
    std::unordered_map<std::string, at::Tensor> attribute_storage_;
};

AppendResult event_stream_into(
    SequenceBuffer& output, const at::Tensor& selected_rows,
    const at::Tensor& identity_ids, const at::Tensor& offsets,
    const Vocabulary& vocabulary,
    const std::unordered_map<std::string, at::Tensor>& attributes = {},
    std::optional<at::Tensor> status_buffer = {});

AppendResult ranked_feature_into(
    SequenceBuffer& output, const at::Tensor& selected_rows,
    const at::Tensor& values, const at::Tensor& feature_ids,
    const Vocabulary& vocabulary, const RankedFeatureOptions& options = {},
    const std::unordered_map<std::string, at::Tensor>& attributes = {},
    std::optional<at::Tensor> status_buffer = {});

} // namespace cellerator::tokenization
