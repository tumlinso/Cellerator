#pragma once

// Benchmark-only cold direct control. This intentionally stores canonical
// interaction records and ordered terms without building packet dictionaries
// or postings. Do not use it with the shared stable fallback.

#include <Cellerator/layout/factorized_interaction.hh>
#include <Cellerator/modeling/cuda/factorized_interaction.hh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cellerator::bench::factorized_interactions::controls {

namespace detail {

inline void validate_direct_extents(
    const layout::FactorizedInteractionExtents& extents) {
    const auto validate = [](std::uint64_t value, const char* what) {
        if (value >= static_cast<std::uint64_t>(
                         std::numeric_limits<std::size_t>::max()))
            throw std::overflow_error(what);
    };
    validate(extents.inputs, "direct input extent exceeds host size");
    validate(extents.contexts, "direct context extent exceeds host size");
    validate(extents.outputs, "direct output extent exceeds host size");
    validate(extents.alpha_rows, "direct alpha extent exceeds host size");
    validate(extents.biases, "direct bias extent exceeds host size");
    validate(extents.gammas, "direct gamma extent exceeds host size");
}

inline void increment(std::unordered_map<std::uint64_t, std::uint64_t>& counts,
                      std::uint64_t key) {
    auto& count = counts[key];
    if (count == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("direct fan-in count overflows uint64");
    ++count;
}

inline std::uint64_t maximum_count(
    const std::unordered_map<std::uint64_t, std::uint64_t>& counts) noexcept {
    std::uint64_t result = 0;
    for (const auto& entry : counts) result = std::max(result, entry.second);
    return result;
}

} // namespace detail

class MinimalDirectHostLayout final {
public:
    // Packet-size options are retained for identity/config reporting but do
    // not constrain this canonical-only representation.
    MinimalDirectHostLayout(
        const std::vector<layout::FactorizedInteractionMechanism>& mechanisms,
        layout::FactorizedInteractionExtents extents,
        layout::FactorizedInteractionOptions options = {})
        : extents_(extents), options_(options) {
        detail::validate_direct_extents(extents_);
        if (!options_.pattern_id || !options_.structure_id ||
            !options_.structure_epoch)
            throw std::invalid_argument(
                "direct control structure identities and epoch must be nonzero");
        if (mechanisms.size() >
            static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()))
            throw std::overflow_error("direct mechanism count exceeds uint64");

        records_.reserve(mechanisms.size());
        std::unordered_set<std::uint64_t> logical_ids;
        std::unordered_map<std::uint64_t, std::uint64_t> output_counts;
        std::unordered_map<std::uint64_t, std::uint64_t> input_role_counts;
        std::unordered_map<std::uint64_t, std::uint64_t> context_term_counts;
        std::unordered_map<std::uint64_t, std::uint64_t> alpha_counts;
        std::unordered_map<std::uint64_t, std::uint64_t> bias_counts;
        std::unordered_map<std::uint64_t, std::uint64_t> gamma_counts;

        std::uint64_t term_begin = 0;
        for (std::size_t ordinal = 0; ordinal < mechanisms.size(); ++ordinal) {
            const auto& source = mechanisms[ordinal];
            if (!source.logical_id || !logical_ids.insert(source.logical_id).second)
                throw std::invalid_argument(
                    "direct mechanism logical IDs must be nonzero and unique");
            if (source.arity < 2 ||
                source.arity > layout::factorized_interaction_max_arity)
                throw std::invalid_argument("direct mechanism arity must be 2, 3, or 4");
            if (source.terms.size() >
                static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
                throw std::overflow_error(
                    "direct mechanism term count exceeds uint32 record width");
            if (source.target >= extents_.outputs ||
                source.alpha >= extents_.alpha_rows ||
                source.bias >= extents_.biases)
                throw std::out_of_range(
                    "direct mechanism target or parameter index exceeds its extent");
            if (term_begin > std::numeric_limits<std::uint64_t>::max() -
                                 source.terms.size())
                throw std::overflow_error("direct term count overflows uint64");
            if (source.terms.size() >
                std::numeric_limits<std::size_t>::max() - terms_.size())
                throw std::overflow_error("direct term storage exceeds host size");

            layout::FactorizedInteractionRecord record{};
            record.logical_id = source.logical_id;
            record.target = source.target;
            record.alpha = source.alpha;
            record.bias = source.bias;
            record.term_begin = term_begin;
            record.term_count = static_cast<std::uint32_t>(source.terms.size());
            record.arity = source.arity;
            for (std::uint32_t role = 0; role < source.arity; ++role) {
                if (source.inputs[role] >= extents_.inputs)
                    throw std::out_of_range(
                        "direct input role exceeds its occurrence extent");
                record.inputs[role] = source.inputs[role];
                detail::increment(input_role_counts, source.inputs[role]);
            }
            detail::increment(output_counts, source.target);
            detail::increment(alpha_counts, source.alpha);
            detail::increment(bias_counts, source.bias);
            records_.push_back(record);

            for (const auto& term : source.terms) {
                if (term.context >= extents_.contexts ||
                    term.gamma >= extents_.gammas)
                    throw std::out_of_range(
                        "direct context or gamma index exceeds its extent");
                terms_.push_back({term.context, term.gamma,
                                  static_cast<std::uint64_t>(ordinal)});
                detail::increment(context_term_counts, term.context);
                detail::increment(gamma_counts, term.gamma);
            }
            term_begin += source.terms.size();
        }

        max_output_mechanisms_ = detail::maximum_count(output_counts);
        max_input_roles_ = detail::maximum_count(input_role_counts);
        max_context_terms_ = detail::maximum_count(context_term_counts);
        max_alpha_mechanisms_ = detail::maximum_count(alpha_counts);
        max_bias_mechanisms_ = detail::maximum_count(bias_counts);
        max_gamma_terms_ = detail::maximum_count(gamma_counts);
        for (const auto& record : records_)
            max_mechanism_terms_ =
                std::max<std::uint64_t>(max_mechanism_terms_, record.term_count);
    }

    // Returned pointers borrow this immutable host owner's vectors.
    layout::FactorizedInteractionView view() const noexcept {
        layout::FactorizedInteractionView result{};
        result.schema = layout::factorized_interaction_schema;
        result.pattern_id = options_.pattern_id;
        result.structure_id = options_.structure_id;
        result.structure_epoch = options_.structure_epoch;
        result.extents = extents_;
        result.mechanism_count = static_cast<std::uint64_t>(records_.size());
        result.term_count = static_cast<std::uint64_t>(terms_.size());
        result.max_output_mechanisms = max_output_mechanisms_;
        result.max_input_roles = max_input_roles_;
        result.max_context_terms = max_context_terms_;
        result.max_alpha_mechanisms = max_alpha_mechanisms_;
        result.max_bias_mechanisms = max_bias_mechanisms_;
        result.max_gamma_terms = max_gamma_terms_;
        result.max_mechanism_terms = max_mechanism_terms_;
        result.mechanisms = records_.empty() ? nullptr : records_.data();
        result.terms = terms_.empty() ? nullptr : terms_.data();
        // packet_count and every packet/posting pointer intentionally remain 0/null.
        return result;
    }

    cudaError_t backend_for(
        modeling::cuda::FactorizedInteractionProblem problem,
        modeling::cuda::FactorizedInteractionBackend* selected) const noexcept {
        return modeling::cuda::factorized_interaction_backend(
            view(), problem, modeling::cuda::FactorizedInteractionRoute::direct,
            selected);
    }

    bool direct_only_applicable(
        modeling::cuda::FactorizedInteractionProblem problem) const noexcept {
        modeling::cuda::FactorizedInteractionBackend selected{};
        return backend_for(problem, &selected) == cudaSuccess &&
               selected == modeling::cuda::FactorizedInteractionBackend::direct;
    }

    const std::vector<layout::FactorizedInteractionRecord>&
    mechanisms() const noexcept { return records_; }
    const std::vector<layout::FactorizedInteractionTermRecord>&
    terms() const noexcept { return terms_; }
    const layout::FactorizedInteractionExtents& extents() const noexcept {
        return extents_;
    }
    const layout::FactorizedInteractionOptions& options() const noexcept {
        return options_;
    }

private:
    layout::FactorizedInteractionExtents extents_{};
    layout::FactorizedInteractionOptions options_{};
    std::vector<layout::FactorizedInteractionRecord> records_;
    std::vector<layout::FactorizedInteractionTermRecord> terms_;
    std::uint64_t max_output_mechanisms_ = 0;
    std::uint64_t max_input_roles_ = 0;
    std::uint64_t max_context_terms_ = 0;
    std::uint64_t max_alpha_mechanisms_ = 0;
    std::uint64_t max_bias_mechanisms_ = 0;
    std::uint64_t max_gamma_terms_ = 0;
    std::uint64_t max_mechanism_terms_ = 0;
};

// This owner exists only for timing/qualifying the cold canonical direct
// control. Construction queues one bulk H2D upload on upload_stream. Same
// stream use is ordered; another stream must call wait_on(). It requires the
// exact problem shape to select the direct atomic backend; a stable backend
// needs the full prepared layout's posting maps. Keep this owner alive until
// its upload and all queued numerical uses complete.
class MinimalDirectDeviceLayout final {
public:
    MinimalDirectDeviceLayout(const MinimalDirectHostLayout& host,
                              modeling::cuda::FactorizedInteractionProblem problem,
                              int device,
                              const memory::cuda::Stream& upload_stream);
    ~MinimalDirectDeviceLayout() noexcept;
    MinimalDirectDeviceLayout(MinimalDirectDeviceLayout&&) noexcept;
    MinimalDirectDeviceLayout& operator=(MinimalDirectDeviceLayout&&) noexcept;
    MinimalDirectDeviceLayout(const MinimalDirectDeviceLayout&) = delete;
    MinimalDirectDeviceLayout& operator=(const MinimalDirectDeviceLayout&) = delete;

    const layout::FactorizedInteractionView& view() const noexcept;
    bool accepts(modeling::cuda::FactorizedInteractionProblem problem) const noexcept;
    int device() const noexcept;
    std::size_t bytes() const noexcept;
    void wait_on(const memory::cuda::Stream& stream) const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace cellerator::bench::factorized_interactions::controls
