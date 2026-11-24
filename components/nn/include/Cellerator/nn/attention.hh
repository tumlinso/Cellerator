#pragma once

#include <Cellerator/nn/layers.hh>
#include <Cellerator/nn/relations.hh>
#include <Cellerator/nn/types.hh>

#include <cstdint>
#include <span>

namespace cellerator::nn {

enum class AttentionKernel { aten, cellerator_csr };

struct AttentionOptions {
    std::int64_t width = 0;
    std::int64_t heads = 1;
    double dropout = 0.0;
    AttentionKernel kernel = AttentionKernel::aten;
    bool causal = false;
};

// Borrowed host boundaries, read during this call only. Empty boundaries mean
// one packed group; rank-three tensors already declare dense batch groups.
struct AttentionLayout {
    std::span<const std::int64_t> source_offsets{};
    std::span<const std::int64_t> target_offsets{};
    bool causal = false;
};

// Tensor handles refer to the explicit parameter owner. Functional calls can
// borrow ordinary Python/Torch-owned parameters without registering a module.
struct AttentionParameters {
    LinearParameters query;
    LinearParameters key;
    LinearParameters value;
    LinearParameters output;

    AttentionParameters() = default;
    AttentionParameters(ParameterGroup group, std::int64_t width);
};

// Projected relation-restricted attention, used by both language frontends.
// q[Q,H,D], k[N,H,D], v[N,H,V]; rows/columns are validated device occurrence
// indices from prepared support or an explicitly validated raw CSR path.
at::Tensor sparse_attention(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& rows, const at::Tensor& columns,
    double scale, double dropout = 0.0, Run run = {});

at::Tensor relation_attention(
    const at::Tensor& source, const at::Tensor& target,
    const AttentionParameters& parameters, const AttentionOptions& options,
    AttentionLayout layout = {}, Run run = {});

// The prepared support supplies declared occurrence order, grouping, and
// owned structural buffers. Call validate_for with the current endpoint tokens
// at a boundary whenever payloads may have changed order; extents alone cannot
// infer the biological order of an arbitrary Tensor.
at::Tensor relation_attention(
    const at::Tensor& source, const at::Tensor& target,
    const AttentionParameters& parameters, const AttentionOptions& options,
    const PreparedRelation& relation, Run run = {});

class RelationAttention {
public:
    RelationAttention(ParameterGroup group, AttentionOptions options);
    RelationAttention(AttentionParameters parameters, AttentionOptions options);

    at::Tensor operator()(const at::Tensor& source,
                          AttentionLayout layout = {}, Run run = {}) const;
    at::Tensor operator()(const at::Tensor& source, const at::Tensor& target,
                          AttentionLayout layout = {}, Run run = {}) const;
    at::Tensor operator()(const at::Tensor& source,
                          const PreparedRelation& relation, Run run = {}) const;
    at::Tensor operator()(const at::Tensor& source, const at::Tensor& target,
                          const PreparedRelation& relation, Run run = {}) const;

    template<class Support>
    at::Tensor cross(const at::Tensor& source, const at::Tensor& target,
                     const Support& support, Run run = {}) const {
        return (*this)(source, target, support, run);
    }

    const AttentionOptions& options() const noexcept { return options_; }
    const AttentionParameters& parameters() const noexcept { return parameters_; }

private:
    AttentionOptions options_;
    AttentionParameters parameters_;
};

} // namespace cellerator::nn
