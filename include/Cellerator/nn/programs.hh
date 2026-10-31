#pragma once

#include <Cellerator/nn/layers.hh>
#include <Cellerator/nn/parameters.hh>
#include <Cellerator/nn/types.hh>
#include <Cellerator/nn/attention.hh>
#include <Cellerator/nn/relations.hh>
#include <Cellerator/biology/coordinates.hh>

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <utility>

namespace cellerator::nn {

// Learned interaction updates and the corresponding shared program features.
// Both tensors retain occurrence order; callers choose residual composition.
struct ProgramResult {
    at::Tensor feature_updates;
    at::Tensor program_features;
};

// Per-call program selection and optional structural/context inputs. The
// program_bank_indices entries map target occurrences to rows in the learned
// query bank. An absent view selects each bank row once per declared group;
// an explicitly empty view represents no program occurrences. The supplied
// program order token describes those same occurrences; the mapping from token
// identities to bank rows remains caller-owned. structural_generation, when
// supplied, is checked against the prepared relation's declaration.
// Offsets and views are borrowed for the duration of operator().
struct ProgramCall {
    const PreparedRelation* relation = nullptr;
    std::optional<biology::OccurrenceOrderToken> feature_order;
    std::optional<biology::OccurrenceOrderToken> program_order;
    std::span<const std::int64_t> feature_offsets{};
    std::span<const std::int64_t> program_offsets{};
    std::optional<std::span<const std::int64_t>> program_bank_indices;
    std::optional<std::uint64_t> structural_generation;
    at::Tensor context;
};

// Shared learned queries attend to feature occurrences and return contextual
// updates through dense attention or a caller-prepared overlapping relation.
// Program membership and identity remain caller declared; this type does not
// infer a hard topology or causal meaning.
class InteractionPrograms {
public:
    InteractionPrograms(ParameterGroup group, std::int64_t width,
                        std::int64_t num_programs, std::int64_t heads = 1,
                        double dropout = 0.0,
                        std::optional<std::int64_t> context_size = std::nullopt);

    ProgramResult operator()(const at::Tensor& features, Run run = {}) const;
    ProgramResult operator()(const at::Tensor& features,
                             const ProgramCall& call, Run run = {}) const;

    std::int64_t width() const noexcept { return width_; }
    std::int64_t num_programs() const noexcept { return num_programs_; }
    const at::Tensor& program_queries() const noexcept { return program_queries_; }
    ParameterId program_queries_id() const noexcept { return program_queries_id_; }

private:
    ProgramResult apply(const at::Tensor& features, const ProgramCall& call,
                        Run run) const;

    ParameterGroup group_;
    std::int64_t width_ = 0;
    std::int64_t num_programs_ = 0;
    std::optional<std::int64_t> context_size_;
    ParameterId program_queries_id_ = invalid_parameter_id;
    at::Tensor program_queries_;
    std::optional<Linear> context_projection_;
    RelationAttention feature_to_program_;
    RelationAttention program_to_feature_;
};

// Independently gates supplied participant occurrences, sums their encoded
// features (optionally normalizing by gate mass), and applies a joint map.
// A configured context projection conditions participant features. A separate
// callable overload can add a learned/contextual contribution to gate logits.
class GatedInteractor {
public:
    GatedInteractor(ParameterGroup group, std::int64_t input_size,
                    std::int64_t hidden_size, std::int64_t output_size,
                    bool normalize = false,
                    std::optional<std::int64_t> context_size = std::nullopt);

    at::Tensor operator()(const at::Tensor& participants, Run run = {}) const;
    at::Tensor operator()(const at::Tensor& participants,
                          const at::Tensor& context, Run run = {}) const;

    template<class Context, class Conditioner>
    at::Tensor operator()(const at::Tensor& participants, const Context& context,
                          Conditioner&& conditioner, Run run = {}) const {
        return apply_gate_conditioner(
            participants,
            std::invoke(std::forward<Conditioner>(conditioner), context), run);
    }

    std::int64_t input_size() const noexcept { return input_size_; }
    std::int64_t output_size() const noexcept { return output_size_; }
    bool normalizes() const noexcept { return normalize_; }

    const Linear& encoder() const noexcept { return encode_; }
    const Linear& gate() const noexcept { return gate_; }
    const Linear& joint_hidden() const noexcept { return joint_hidden_; }
    const Linear& joint_output() const noexcept { return joint_output_; }

private:
    at::Tensor apply(const at::Tensor& participants, const at::Tensor& context,
                     const at::Tensor& gate_logit_addition, Run run) const;
    at::Tensor apply_gate_conditioner(const at::Tensor& participants,
                                      const at::Tensor& gate_logit_addition,
                                      Run run) const;

    ParameterGroup group_;
    std::int64_t input_size_ = 0;
    std::int64_t output_size_ = 0;
    bool normalize_ = false;
    std::optional<std::int64_t> context_size_;
    Linear encode_;
    Linear gate_;
    std::optional<Linear> context_projection_;
    Linear joint_hidden_;
    Linear joint_output_;
};

} // namespace cellerator::nn

