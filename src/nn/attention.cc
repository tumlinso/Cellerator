#include <Cellerator/nn/attention.hh>
#include <ATen/Functions.h>

#ifdef CELLERATOR_NN_HAS_CUDA
#include <Cellerator/nn/ops.hh>
#endif

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cellerator::nn {
namespace {

void validate_options(const AttentionOptions& o) {
    if (o.width <= 0 || o.heads <= 0 || o.width % o.heads)
        throw std::invalid_argument("attention width must be positive and divisible by heads");
    if (!(o.dropout >= 0.0 && o.dropout < 1.0))
        throw std::invalid_argument("attention dropout must be in [0,1)");
}

void validate_inputs(const at::Tensor& source, const at::Tensor& target,
                     const AttentionOptions& o) {
    validate_options(o);
    if (!source.defined() || !target.defined() ||
        (source.dim() != 2 && source.dim() != 3) || source.dim() != target.dim())
        throw std::invalid_argument("attention expects matching packed rank-2 or dense rank-3 tensors");
    if (source.size(-1) != o.width || target.size(-1) != o.width)
        throw std::invalid_argument("attention payload width mismatch");
    if (source.device() != target.device() || source.scalar_type() != target.scalar_type())
        throw std::invalid_argument("attention source/target must share device and dtype");
    if (!source.is_floating_point())
        throw std::invalid_argument("attention payloads must be floating point");
    if (source.dim() == 3 && source.size(0) != target.size(0))
        throw std::invalid_argument("dense attention source/target batch sizes differ");
}

std::vector<std::int64_t> boundaries(std::span<const std::int64_t> offsets,
                                    std::int64_t extent) {
    if (offsets.empty()) return {0, extent};
    if (offsets.front() != 0 || offsets.back() != extent)
        throw std::invalid_argument("packed offsets must start at zero and end at extent");
    std::int64_t previous = 0;
    for (auto value : offsets) {
        if (value < previous || value > extent)
            throw std::invalid_argument("packed offsets must be nondecreasing and in bounds");
        previous = value;
    }
    return {offsets.begin(), offsets.end()};
}

at::Tensor project(const at::Tensor& x, const LinearParameters& p,
                   const AttentionOptions& o) {
    auto result = linear(x, p);
    auto shape = result.sizes().vec();
    shape.pop_back();
    shape.push_back(o.heads);
    shape.push_back(o.width / o.heads);
    return result.reshape(shape);
}

at::Tensor zero_interaction(const at::Tensor& q, const at::Tensor& k,
                            const at::Tensor& v) {
    // Multiplication before reduction also avoids overflow of unused inputs.
    return at::nan_to_num(q) * 0.0 + (at::nan_to_num(k) * 0.0).sum()
           + (at::nan_to_num(v) * 0.0).sum();
}

at::Tensor dense_heads(const at::Tensor& q, const at::Tensor& k,
                       const at::Tensor& v, double dropout, bool causal) {
    if (q.size(-2) == 0 || k.size(-2) == 0 || q.size(0) == 0)
        return zero_interaction(q, k, v);
    return at::scaled_dot_product_attention(q, k, v, std::nullopt, dropout, causal);
}

} // namespace

AttentionParameters::AttentionParameters(ParameterGroup group, std::int64_t width)
    : query(group, width, width), key(group, width, width),
      value(group, width, width), output(group, width, width, false) {}

at::Tensor relation_attention(const at::Tensor& source, const at::Tensor& target,
    const AttentionParameters& p, const AttentionOptions& o,
    AttentionLayout layout, Run run) {
    validate_inputs(source, target, o);
    if (o.kernel != AttentionKernel::aten)
        throw std::invalid_argument("Cellerator CSR attention requires an explicit PreparedRelation");
    const bool causal = o.causal || layout.causal;
    if (causal && !source.is_same(target))
        throw std::invalid_argument("causal attention requires self-attention");
    const double dropout = run.training ? o.dropout : 0.0;
    auto q = project(target, p.query, o);
    auto k = project(source, p.key, o);
    auto v = project(source, p.value, o);
    at::Tensor result;
    if (source.dim() == 3) {
        if (!layout.source_offsets.empty() || !layout.target_offsets.empty())
            throw std::invalid_argument("host group offsets apply only to packed attention");
        auto heads = dense_heads(q.transpose(1, 2), k.transpose(1, 2),
                                 v.transpose(1, 2), dropout, causal);
        result = heads.transpose(1, 2).reshape({target.size(0), target.size(1), o.width});
    } else {
        auto src = boundaries(layout.source_offsets, source.size(0));
        auto dst = boundaries(layout.target_offsets, target.size(0));
        if (src.size() != dst.size())
            throw std::invalid_argument("packed attention source/target group counts differ");
        std::vector<at::Tensor> groups;
        groups.reserve(src.size() - 1);
        for (std::size_t i = 0; i + 1 < src.size(); ++i) {
            auto qq = q.narrow(0, dst[i], dst[i + 1] - dst[i]).transpose(0, 1).unsqueeze(0);
            auto kk = k.narrow(0, src[i], src[i + 1] - src[i]).transpose(0, 1).unsqueeze(0);
            auto vv = v.narrow(0, src[i], src[i + 1] - src[i]).transpose(0, 1).unsqueeze(0);
            auto heads = dense_heads(qq, kk, vv, dropout, causal);
            groups.push_back(heads.squeeze(0).transpose(0, 1).reshape({dst[i + 1] - dst[i], o.width}));
        }
        result = groups.empty() ? zero_interaction(q, k, v).reshape({target.size(0), o.width})
                                : at::cat(groups, 0);
    }
    return linear(result, p.output);
}

at::Tensor relation_attention(const at::Tensor& source, const at::Tensor& target,
    const AttentionParameters& p, const AttentionOptions& o,
    const PreparedRelation& relation, Run run) {
    validate_inputs(source, target, o);
    relation.validate_storage();
    const auto source_count = source.numel() / o.width;
    const auto target_count = target.numel() / o.width;
    if (source_count != relation.source_count() || target_count != relation.target_count()
        || source.device() != relation.device())
        throw std::invalid_argument("prepared relation payload extent/device mismatch");
    if (source.dim() == 3) {
        const auto& src_offsets = relation.source_offsets();
        const auto& dst_offsets = relation.target_offsets();
        const auto groups = source.size(0);
        if (src_offsets.size() != static_cast<std::size_t>(groups + 1)
            || dst_offsets.size() != static_cast<std::size_t>(groups + 1))
            throw std::invalid_argument("prepared relation groups must match dense batch groups");
        for (std::int64_t group = 0; group <= groups; ++group) {
            if (src_offsets[group] != group * source.size(1)
                || dst_offsets[group] != group * target.size(1))
                throw std::invalid_argument("prepared relation offsets disagree with dense batch layout");
        }
    }
    if (o.causal && !source.is_same(target))
        throw std::invalid_argument("causal attention requires self-attention");
    auto q = project(target.reshape({target_count, o.width}), p.query, o);
    auto k = project(source.reshape({source_count, o.width}), p.key, o);
    auto v = project(source.reshape({source_count, o.width}), p.value, o);
    const double dropout = run.training ? o.dropout : 0.0;
    at::Tensor combined;
    if (o.kernel == AttentionKernel::cellerator_csr) {
        if (o.causal && relation.has_future_edges())
            throw std::invalid_argument("native causal attention requires explicitly prepared relation.causal() support");
#ifdef CELLERATOR_NN_HAS_CUDA
        combined = ops::csr_scaled_dot_product_attention_cuda_prepared(
            q, k, v, relation.row_ptr(), relation.col_indices(),
            1.0 / std::sqrt(static_cast<double>(o.width / o.heads)), dropout);
#else
        throw std::runtime_error("native CSR attention requires the optional Cellerator::nn_cuda component");
#endif
    } else {
        auto rows = relation.rows();
        auto columns = relation.col_indices();
        if (o.causal) {
            rows = rows.index({relation.causal_mask()});
            columns = columns.index({relation.causal_mask()});
        }
        if (columns.numel() == 0) {
            combined = zero_interaction(q, k, v);
        } else {
            const auto scale = 1.0 / std::sqrt(static_cast<double>(o.width / o.heads));
            auto scores = (q.index_select(0, rows) * k.index_select(0, columns)).sum(-1) * scale;
            auto row_heads = rows.unsqueeze(-1).expand({rows.numel(), o.heads});
            auto maxima = at::full({target_count, o.heads}, -std::numeric_limits<double>::infinity(), scores.options());
            maxima = at::scatter_reduce(maxima, 0, row_heads, scores, "amax", true);
            auto numerators = (scores - maxima.index_select(0, rows)).exp();
            auto denominator = at::zeros({target_count, o.heads}, scores.options()).index_add(0, rows, numerators);
            auto weights = numerators / denominator.index_select(0, rows);
            if (dropout > 0) weights = at::dropout(weights, dropout, true);
            auto messages = v.index_select(0, columns) * weights.unsqueeze(-1);
            combined = at::zeros({target_count, o.heads, o.width / o.heads}, v.options()).index_add(0, rows, messages);
        }
    }
    auto shape = target.sizes().vec();
    return linear(combined.reshape({target_count, o.width}), p.output).reshape(shape);
}

RelationAttention::RelationAttention(ParameterGroup group, AttentionOptions options)
    : options_(options), parameters_(group, options.width) { validate_options(options_); }
RelationAttention::RelationAttention(AttentionParameters parameters, AttentionOptions options)
    : options_(options), parameters_(std::move(parameters)) { validate_options(options_); }
at::Tensor RelationAttention::operator()(const at::Tensor& x, AttentionLayout layout, Run run) const {
    return relation_attention(x, x, parameters_, options_, layout, run);
}
at::Tensor RelationAttention::operator()(const at::Tensor& x, const at::Tensor& y, AttentionLayout layout, Run run) const {
    return relation_attention(x, y, parameters_, options_, layout, run);
}
at::Tensor RelationAttention::operator()(const at::Tensor& x, const PreparedRelation& relation, Run run) const {
    return relation_attention(x, x, parameters_, options_, relation, run);
}
at::Tensor RelationAttention::operator()(const at::Tensor& x, const at::Tensor& y, const PreparedRelation& relation, Run run) const {
    return relation_attention(x, y, parameters_, options_, relation, run);
}

} // namespace cellerator::nn
