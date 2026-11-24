#include <Cellerator/compute/tokenization/tokenization.hh>

#include <Cellerator/compute/operators/aten_ops.hh>

#include <ATen/ATen.h>
#include <ATen/ops/bincount.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/searchsorted.h>
#include <ATen/ops/sort.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace cellerator::tokenization {
namespace {

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

bool scalar_bool(const at::Tensor& value) {
    return value.item<bool>();
}

void validate_offsets(const at::Tensor& ids, const at::Tensor& offsets) {
    require(ids.defined() && ids.scalar_type() == at::kLong && ids.dim() == 1,
            "identity/token IDs must be a rank-1 int64 tensor");
    require(offsets.defined() && offsets.scalar_type() == at::kLong
            && offsets.dim() == 1 && offsets.numel() > 0,
            "offsets must be a nonempty rank-1 int64 tensor");
    require(offsets.device() == ids.device(), "IDs and offsets must share a device");
    require(offsets.index({0}).item<std::int64_t>() == 0
            && offsets.index({-1}).item<std::int64_t>() == ids.numel(),
            "offsets must start at zero and end at the ID count");
    if (offsets.numel() > 1)
        require(scalar_bool((offsets.slice(0, 1) >= offsets.slice(0, 0,
                    offsets.numel() - 1)).all()),
                "offsets must be nondecreasing");
}

void validate_attributes(const std::unordered_map<std::string, at::Tensor>& attrs,
                         std::int64_t occurrences, const at::Device& device) {
    for (const auto& [name, value] : attrs) {
        require(!name.empty(), "attribute names must not be empty");
        require(value.defined() && value.dim() >= 1 && value.size(0) == occurrences,
                "attribute leading extent must match token occurrences");
        require(value.device() == device, "attributes and token IDs must share a device");
    }
}

at::Tensor offsets_after_filter(const at::Tensor& offsets, const at::Tensor& keep) {
    const auto n = keep.numel();
    const auto rows_n = offsets.numel() - 1;
    auto positions = at::arange(n, offsets.options());
    auto row_for_position = at::searchsorted(offsets.slice(0, 1), positions,
                                              false, true);
    auto retained_rows = row_for_position.masked_select(keep);
    auto counts = at::bincount(retained_rows, {}, rows_n);
    auto zero = at::zeros({1}, offsets.options());
    return at::cat({zero, at::cumsum(counts, 0)});
}

at::Tensor masked_attribute(const at::Tensor& values, const at::Tensor& keep) {
    auto positions = at::nonzero(keep).reshape({-1});
    return values.index_select(0, positions);
}

at::Tensor ranked_columns(const at::Tensor& matrix, const at::Tensor& flat_indices) {
    const auto columns = matrix.size(1);
    if (matrix.layout() == at::kSparseCsr)
        return matrix.col_indices().index_select(0, flat_indices);
    return flat_indices.remainder(columns);
}

at::Tensor safe_factor(const at::Tensor& factors) {
    auto valid = at::isfinite(factors) & (factors > 0);
    return at::where(valid, factors, at::ones_like(factors));
}

at::Tensor vocabulary_membership(const at::Tensor& identities, const Vocabulary& vocabulary) {
    if (identities.numel() == 0)
        return at::empty({0}, identities.options().dtype(at::kBool));
    if (vocabulary.identities.numel() == 0)
        return at::zeros(identities.sizes(), identities.options().dtype(at::kBool));
    auto slots = at::searchsorted(vocabulary.identities, identities, false, false);
    auto in_bounds = slots < vocabulary.identities.numel();
    auto safe_slots = slots.clamp(0, vocabulary.identities.numel() - 1);
    return in_bounds & vocabulary.identities.index_select(0, safe_slots).eq(identities);
}

} // namespace

Vocabulary make_vocabulary(const at::Tensor& identities, const at::Tensor& token_ids,
                           unknown_identity_policy policy,
                           std::optional<std::int64_t> unknown_token_id) {
    require(identities.defined() && identities.scalar_type() == at::kLong
            && identities.dim() == 1, "vocabulary identities must be rank-1 int64");
    require(token_ids.defined() && token_ids.scalar_type() == at::kLong
            && token_ids.dim() == 1 && token_ids.numel() == identities.numel(),
            "vocabulary token IDs must be aligned rank-1 int64");
    require(identities.device() == token_ids.device(),
            "vocabulary identities and token IDs must share a device");
    if (identities.numel() > 1) {
        const auto ordered = identities.slice(0, 1) > identities.slice(0, 0,
                                                                      identities.numel() - 1);
        require(scalar_bool(ordered.all()), "vocabulary identities must be sorted and unique");
    }
    require(scalar_bool((token_ids >= 0).all()), "vocabulary token IDs must be nonnegative");
    require((policy == unknown_identity_policy::map_to_token) == unknown_token_id.has_value(),
            "unknown token ID must be specified only for map_to_token policy");
    if (unknown_token_id)
        require(*unknown_token_id >= 0, "unknown token ID must be nonnegative");
    return {identities, token_ids, policy, unknown_token_id};
}

Vocabulary Vocabulary::to(at::Device device) const {
    return {identities.to(device), token_ids.to(device), unknown_policy, unknown_token_id};
}

void validate_token_batch(const TokenBatch& batch) {
    validate_offsets(batch.token_ids, batch.offsets);
    validate_attributes(batch.attributes, batch.token_ids.numel(), batch.token_ids.device());
    require(scalar_bool((batch.token_ids >= 0).all()), "token IDs must be nonnegative");
}

TokenBatch map_identities(const at::Tensor& identity_ids, const at::Tensor& offsets,
                          const Vocabulary& vocabulary,
                          const std::unordered_map<std::string, at::Tensor>& attributes) {
    validate_offsets(identity_ids, offsets);
    validate_attributes(attributes, identity_ids.numel(), identity_ids.device());
    require(vocabulary.identities.defined() && vocabulary.token_ids.defined()
            && vocabulary.identities.scalar_type() == at::kLong
            && vocabulary.token_ids.scalar_type() == at::kLong,
            "vocabulary is not initialized");
    require(identity_ids.device() == vocabulary.identities.device()
            && identity_ids.device() == vocabulary.token_ids.device(),
            "identity IDs and prepared vocabulary must share a device");

    const auto vocab_size = vocabulary.identities.numel();
    if (vocab_size == 0 && identity_ids.numel() != 0
        && vocabulary.unknown_policy == unknown_identity_policy::error)
        throw std::invalid_argument("identity is absent from empty vocabulary");

    at::Tensor known;
    at::Tensor tokens;
    if (identity_ids.numel() == 0) {
        known = at::empty({0}, identity_ids.options().dtype(at::kBool));
        tokens = at::empty({0}, identity_ids.options());
    } else if (vocab_size == 0) {
        known = at::zeros({identity_ids.numel()}, identity_ids.options().dtype(at::kBool));
        tokens = at::zeros_like(identity_ids);
    } else {
        auto slots = at::searchsorted(vocabulary.identities, identity_ids, false, false);
        auto in_bounds = slots < vocab_size;
        auto safe_slots = slots.clamp(0, vocab_size - 1);
        known = in_bounds & vocabulary.identities.index_select(0, safe_slots).eq(identity_ids);
        tokens = vocabulary.token_ids.index_select(0, safe_slots);
    }

    if (vocabulary.unknown_policy == unknown_identity_policy::error) {
        require(scalar_bool(known.all()), "identity is absent from vocabulary");
        return {tokens, offsets, attributes};
    }
    if (vocabulary.unknown_policy == unknown_identity_policy::map_to_token)
        tokens = at::where(known, tokens,
            at::full_like(tokens, *vocabulary.unknown_token_id));

    if (vocabulary.unknown_policy == unknown_identity_policy::drop) {
        auto ids = at::nonzero(known).reshape({-1});
        std::unordered_map<std::string, at::Tensor> filtered;
        for (const auto& [name, value] : attributes)
            filtered.emplace(name, value.index_select(0, ids));
        return {tokens.index_select(0, ids), offsets_after_filter(offsets, known),
                std::move(filtered)};
    }
    return {tokens, offsets, attributes};
}

TokenBatch event_stream(const at::Tensor& identity_ids, const at::Tensor& offsets,
                        const Vocabulary& vocabulary,
                        const std::unordered_map<std::string, at::Tensor>& attributes) {
    return map_identities(identity_ids, offsets, vocabulary, attributes);
}

TokenBatch add_boundaries(const TokenBatch& batch,
                          const std::optional<TokenBatch>& prefix,
                          const std::optional<TokenBatch>& suffix) {
    validate_token_batch(batch);
    const auto rows = batch.offsets.numel() - 1;
    const auto device = batch.token_ids.device();
    auto validate_boundary = [&](const TokenBatch& boundary) {
        validate_token_batch(boundary);
        require(boundary.token_ids.device() == device
                && boundary.offsets.numel() == rows + 1,
                "boundary batch must be aligned with sequence count and device");
        auto length = boundary.offsets.slice(0, 1) - boundary.offsets.slice(0, 0, rows);
        require(scalar_bool((length <= 1).all()),
                "a prefix or suffix supplies at most one token per sequence");
        require(boundary.attributes.size() == batch.attributes.size(),
                "boundary and sequence attribute schemas must match");
        for (const auto& [name, value] : boundary.attributes) {
            auto it = batch.attributes.find(name);
            require(it != batch.attributes.end() && it->second.scalar_type() == value.scalar_type()
                    && it->second.sizes().slice(1) == value.sizes().slice(1),
                    "boundary attributes must match token attribute schema");
        }
    };
    if (prefix) validate_boundary(*prefix);
    if (suffix) validate_boundary(*suffix);

    std::vector<const TokenBatch*> parts;
    if (prefix) parts.push_back(&*prefix);
    parts.push_back(&batch);
    if (suffix) parts.push_back(&*suffix);
    auto lengths = at::zeros({rows}, batch.offsets.options());
    for (const auto* part : parts)
        lengths = lengths + part->offsets.slice(0, 1) - part->offsets.slice(0, 0, rows);
    auto offsets = at::cat({at::zeros({1}, batch.offsets.options()), at::cumsum(lengths, 0)});
    const auto total = offsets.index({-1}).item<std::int64_t>();
    auto ids = at::empty({total}, batch.token_ids.options());
    std::unordered_map<std::string, at::Tensor> attrs;
    for (const auto& [name, value] : batch.attributes) {
        std::vector<std::int64_t> shape{total};
        for (std::int64_t d = 1; d < value.dim(); ++d) shape.push_back(value.size(d));
        attrs.emplace(name, at::empty(shape, value.options()));
    }
    auto prior = at::zeros({rows}, batch.offsets.options());
    for (const auto* part : parts) {
        if (part->token_ids.numel() > 0) {
            auto source_pos = at::arange(part->token_ids.numel(), batch.offsets.options());
            auto source_row = at::searchsorted(part->offsets.slice(0, 1), source_pos, false, true);
            auto source_start = part->offsets.index_select(0, source_row);
            auto dest = offsets.index_select(0, source_row) + prior.index_select(0, source_row)
                        + source_pos - source_start;
            ids.index_copy_(0, dest, part->token_ids);
            for (const auto& [name, value] : part->attributes)
                attrs.at(name).reshape({total, -1}).index_copy_(
                    0, dest, value.reshape({value.size(0), -1}));
        }
        prior = prior + part->offsets.slice(0, 1) - part->offsets.slice(0, 0, rows);
    }
    TokenBatch result{ids, offsets, std::move(attrs)};
    validate_token_batch(result);
    return result;
}

TokenBatch ranked_feature(const at::Tensor& values, const at::Tensor& feature_ids,
                          const Vocabulary& vocabulary,
                          const RankedFeatureOptions& options,
                          const std::unordered_map<std::string, at::Tensor>& attributes) {
    aten_ops::validate_feature_matrix(values);
    require(feature_ids.defined() && feature_ids.scalar_type() == at::kLong
            && feature_ids.dim() == 1 && feature_ids.numel() == values.size(1),
            "feature_ids must be int64 and aligned to matrix columns");
    require(feature_ids.device() == values.device(),
            "feature IDs and values must share a device");
    require(vocabulary.identities.device() == values.device(),
            "feature IDs and vocabulary must share a device");
    require(options.max_per_row >= -1, "max_per_row must be -1 or nonnegative");
    if (feature_ids.numel() > 1) {
        auto sorted_ids = std::get<0>(at::sort(feature_ids));
        require(scalar_bool((sorted_ids.slice(0, 1)
                  > sorted_ids.slice(0, 0, sorted_ids.numel() - 1)).all()),
                "feature IDs must be unique");
    }

    const bool sparse = values.layout() == at::kSparseCsr;
    at::Tensor flat_values;
    at::Tensor eligible;
    if (sparse) {
        flat_values = values.values();
        eligible = flat_values.ne(0);
        if (options.eligibility) {
            require(options.eligibility->scalar_type() == at::kBool
                    && options.eligibility->dim() == 1
                    && options.eligibility->numel() == flat_values.numel()
                    && options.eligibility->device() == values.device(),
                    "CSR eligibility must be bool[nnz]");
            eligible = eligible & *options.eligibility;
        }
        if (options.scores) {
            require(options.scores->layout() == at::kSparseCsr
                    && options.scores->sizes() == values.sizes()
                    && at::equal(options.scores->crow_indices(), values.crow_indices())
                    && at::equal(options.scores->col_indices(), values.col_indices()),
                    "CSR scores must use the same canonical structure as values");
        }
    } else {
        flat_values = values.reshape({-1});
        eligible = flat_values.ne(0);
        if (options.eligibility) {
            require(options.eligibility->scalar_type() == at::kBool
                    && options.eligibility->sizes() == values.sizes()
                    && options.eligibility->device() == values.device(),
                    "dense eligibility must be bool[B,F]");
            eligible = eligible & options.eligibility->reshape({-1});
        }
    }

    if (vocabulary.unknown_policy == unknown_identity_policy::drop) {
        auto known_features = vocabulary_membership(feature_ids, vocabulary);
        if (sparse)
            eligible = eligible & known_features.index_select(0, values.col_indices());
        else
            eligible = eligible & known_features.unsqueeze(0).expand(values.sizes()).reshape({-1});
    }

    const auto& score_matrix = options.scores ? *options.scores : values;
    if (options.scores)
        require(options.scores->device() == values.device()
                && options.scores->layout() == values.layout()
                && options.scores->sizes() == values.sizes(),
                "scores must match the value matrix layout, shape, and device");
    auto scores = sparse ? score_matrix.values() : score_matrix.reshape({-1});
    auto selection_eligibility = sparse ? eligible : eligible.reshape(values.sizes());
    auto selection = aten_ops::select_ranked_features(
        score_matrix, feature_ids, selection_eligibility, options.max_per_row);
    auto selected_positions = selection.flat_indices;
    auto selected_columns = ranked_columns(values, selected_positions);
    auto selected_identity = feature_ids.index_select(0, selected_columns);

    std::unordered_map<std::string, at::Tensor> selected_attributes;
    for (const auto& [name, value] : attributes) {
        const auto expected = sparse ? values.values().numel() : values.numel();
        require(value.defined() && value.dim() >= 1 && value.size(0) == expected
                && value.device() == values.device(),
                "feature attributes must align with dense entries or CSR values");
        selected_attributes.emplace(name, value.index_select(0, selected_positions));
    }
    if (options.retain_measurements) {
        require(selected_attributes.find("measurement") == selected_attributes.end(),
                "measurement attribute name is reserved by retain_measurements");
        selected_attributes.emplace("measurement", flat_values.index_select(0, selected_positions));
    }
    if (options.retain_scores) {
        require(selected_attributes.find("score") == selected_attributes.end(),
                "score attribute name is reserved by retain_scores");
        selected_attributes.emplace("score", scores.index_select(0, selected_positions));
    }
    auto mapped = map_identities(selected_identity, selection.offsets, vocabulary,
                                 selected_attributes);
    validate_token_batch(mapped);
    return mapped;
}

TokenBatch geneformer_style(const at::Tensor& counts, const at::Tensor& total_counts,
                            const at::Tensor& feature_medians, const at::Tensor& feature_ids,
                            const Vocabulary& vocabulary, std::int64_t max_per_row,
                            const std::unordered_map<std::string, at::Tensor>& attributes,
                            std::optional<at::ScalarType> compute_type) {
    aten_ops::validate_feature_matrix(counts);
    const auto rows = counts.size(0);
    const auto columns = counts.size(1);
    require(total_counts.defined() && total_counts.dim() == 1
            && total_counts.numel() == rows && total_counts.device() == counts.device(),
            "total_counts must be a device-aligned vector with one value per row");
    require(feature_medians.defined() && feature_medians.dim() == 1
            && feature_medians.numel() == columns && feature_medians.device() == counts.device(),
            "feature_medians must be device-aligned and match feature columns");
    auto count_values = counts.layout() == at::kSparseCsr ? counts.values() : counts;
    require(scalar_bool(at::isfinite(total_counts).all()),
            "Geneformer-style total counts must be finite");
    require(scalar_bool(((count_values.ge(0)) & at::isfinite(count_values)).all()),
            "Geneformer-style counts must be nonnegative");
    require(max_per_row >= -1, "max_per_row must be -1 or nonnegative");

    const bool sparse = counts.layout() == at::kSparseCsr;
    auto value_entries = sparse ? counts.values() : counts.reshape({-1});
    auto row_nonempty = sparse
        ? at::zeros({rows}, at::TensorOptions().dtype(at::kBool).device(counts.device()))
        : counts.ne(0).any(1);
    if (sparse && counts.values().numel() > 0) {
        auto positive_rows = at::repeat_interleave(
            at::arange(rows, at::TensorOptions().dtype(at::kLong).device(counts.device())),
            counts.crow_indices().slice(0, 1) - counts.crow_indices().slice(0, 0, rows));
        row_nonempty.index_fill_(0, positive_rows.masked_select(counts.values() > 0), true);
    }
    require(scalar_bool(((total_counts > 0) | ~row_nonempty).all()),
            "nonempty rows require positive total counts");
    auto positive_factors = (feature_medians > 0) & at::isfinite(feature_medians);
    auto eligible_features = (vocabulary.unknown_policy == unknown_identity_policy::drop)
        ? vocabulary_membership(feature_ids, vocabulary)
        : at::ones({columns}, feature_medians.options().dtype(at::kBool));
    if (sparse) {
        auto used = at::zeros({columns}, feature_medians.options().dtype(at::kBool));
        if (counts.values().numel() > 0) {
            auto active = counts.values() > 0;
            auto active_cols = counts.col_indices().masked_select(active);
            if (vocabulary.unknown_policy == unknown_identity_policy::drop)
                active_cols = active_cols.masked_select(
                    eligible_features.index_select(0, active_cols));
            used.index_fill_(0, active_cols, true);
        }
        require(scalar_bool((positive_factors | ~used).all()),
                "positive retained features require positive reference medians");
    } else {
        auto used = (counts > 0).any(0) & eligible_features;
        require(scalar_bool((positive_factors | ~used).all()),
                "positive retained features require positive reference medians");
    }

    const auto arithmetic_type = compute_type.value_or(
        counts.scalar_type() == at::kDouble ? at::kDouble : at::kFloat);
    require(arithmetic_type == at::kFloat || arithmetic_type == at::kDouble,
            "Geneformer-style compute_type must be float32 or float64");
    auto compute_counts = counts.to(arithmetic_type);
    auto safe_totals = at::where(total_counts > 0, total_counts,
                                 at::ones_like(total_counts)).to(arithmetic_type);
    auto safe_medians = safe_factor(feature_medians).to(arithmetic_type);
    auto scores = aten_ops::normalize_features(
        compute_counts, safe_totals, safe_medians, 10000.0, arithmetic_type);
    RankedFeatureOptions options;
    options.max_per_row = max_per_row;
    options.scores = scores;
    options.retain_measurements = false;
    options.retain_scores = false;
    auto eligible = sparse ? (value_entries > 0) : (counts > 0);
    options.eligibility = eligible;
    return ranked_feature(counts, feature_ids, vocabulary, options, attributes);
}

PaddedBatch pad_sequences(const TokenBatch& batch, std::int64_t pad_token_id) {
    validate_token_batch(batch);
    require(pad_token_id >= 0, "pad token ID must be nonnegative");
    auto lengths = batch.offsets.slice(0, 1) - batch.offsets.slice(0, 0,
                                                                     batch.offsets.numel() - 1);
    const auto max_length = lengths.numel() == 0 ? 0 : lengths.max().item<std::int64_t>();
    const auto rows = lengths.numel();
    auto padded = at::full({rows, max_length}, pad_token_id, batch.token_ids.options());
    auto valid = at::zeros({rows, max_length}, batch.token_ids.options().dtype(at::kBool));
    if (batch.token_ids.numel() > 0) {
        auto positions = at::arange(batch.token_ids.numel(), batch.offsets.options());
        auto sequence = at::searchsorted(batch.offsets.slice(0, 1), positions, false, true);
        auto starts = batch.offsets.index_select(0, sequence);
        auto within = positions - starts;
        auto flat = sequence * max_length + within;
        padded.view({-1}).index_copy_(0, flat, batch.token_ids);
        valid.view({-1}).index_fill_(0, flat, true);
    }
    std::unordered_map<std::string, at::Tensor> padded_attrs;
    for (const auto& [name, values] : batch.attributes) {
        std::vector<std::int64_t> shape{rows, max_length};
        for (std::int64_t dim = 1; dim < values.dim(); ++dim) shape.push_back(values.size(dim));
        auto out = at::zeros(shape, values.options());
        if (values.numel() > 0)
            out.view({-1, values.numel() / std::max<std::int64_t>(1, batch.token_ids.numel())})
                .index_copy_(0, [&] {
                    auto positions = at::arange(batch.token_ids.numel(), batch.offsets.options());
                    auto sequence = at::searchsorted(batch.offsets.slice(0, 1), positions, false, true);
                    return sequence * max_length + positions - batch.offsets.index_select(0, sequence);
                }(), values.reshape({values.size(0), -1}));
        padded_attrs.emplace(name, std::move(out));
    }
    return {padded, lengths, valid, std::move(padded_attrs)};
}

TokenBatch pack_padded(const at::Tensor& token_ids, const at::Tensor& lengths,
                       const std::unordered_map<std::string, at::Tensor>& padded_attributes) {
    require(token_ids.defined() && token_ids.scalar_type() == at::kLong
            && token_ids.dim() == 2 && scalar_bool((token_ids >= 0).all()),
            "padded token IDs must be a nonnegative int64 matrix");
    require(lengths.defined() && lengths.scalar_type() == at::kLong && lengths.dim() == 1
            && lengths.numel() == token_ids.size(0) && lengths.device() == token_ids.device(),
            "padded lengths must be a device-aligned int64 vector");
    require(scalar_bool(((lengths >= 0) & (lengths <= token_ids.size(1))).all()),
            "padded sequence length exceeds its row extent");
    auto positions = at::arange(token_ids.size(1), token_ids.options());
    auto mask = positions.unsqueeze(0) < lengths.unsqueeze(1);
    auto selected = at::nonzero(mask);
    auto flat_positions = selected.select(1, 0) * token_ids.size(1) + selected.select(1, 1);
    auto values = token_ids.reshape({-1}).index_select(0, flat_positions);
    std::unordered_map<std::string, at::Tensor> attrs;
    for (const auto& [name, attribute] : padded_attributes) {
        require(attribute.dim() >= 2 && attribute.size(0) == token_ids.size(0)
                && attribute.size(1) == token_ids.size(1)
                && attribute.device() == token_ids.device(),
                "padded attributes must align with the first two token dimensions");
        attrs.emplace(name, attribute.reshape({token_ids.size(0) * token_ids.size(1), -1})
                            .index_select(0, flat_positions)
                            .reshape([&] {
                                std::vector<std::int64_t> shape{flat_positions.numel()};
                                for (std::int64_t d = 2; d < attribute.dim(); ++d)
                                    shape.push_back(attribute.size(d));
                                return shape;
                            }()));
    }
    auto offsets = at::cat({at::zeros({1}, lengths.options()), at::cumsum(lengths, 0)});
    TokenBatch result{values, offsets, std::move(attrs)};
    validate_token_batch(result);
    return result;
}

} // namespace cellerator::tokenization
