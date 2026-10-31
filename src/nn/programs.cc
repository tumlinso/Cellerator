#include <Cellerator/nn/programs.hh>

#include <ATen/ATen.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cellerator::nn {
namespace {

std::int64_t checked_positive(std::int64_t value, const char* name) {
    if (value <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return value;
}

std::optional<std::int64_t> checked_context_size(
    std::optional<std::int64_t> context_size) {
    if (context_size && *context_size <= 0) {
        throw std::invalid_argument("context_size must be positive or absent");
    }
    return context_size;
}

std::optional<Linear> make_context_projection(
    ParameterGroup group, std::optional<std::int64_t> context_size,
    std::int64_t output_size) {
    if (!context_size) {
        return std::nullopt;
    }
    return Linear(group, *context_size, output_size, false);
}

void require_floating(const at::Tensor& tensor, const char* name);

std::int64_t checked_program_width(std::int64_t width,
                                   std::int64_t num_programs,
                                   std::int64_t heads, double dropout,
                                   std::optional<std::int64_t> context_size) {
    checked_positive(width, "width");
    checked_positive(num_programs, "num_programs");
    checked_positive(heads, "heads");
    checked_context_size(context_size);
    if (width % heads != 0) {
        throw std::invalid_argument("heads must divide width");
    }
    if (!(dropout >= 0.0 && dropout < 1.0)) {
        throw std::invalid_argument("dropout must be in [0, 1)");
    }
    return width;
}

AttentionOptions program_attention_options(std::int64_t width,
                                           std::int64_t heads,
                                           double dropout) {
    AttentionOptions options;
    options.width = width;
    options.heads = heads;
    options.dropout = dropout;
    options.kernel = AttentionKernel::aten;
    return options;
}

std::vector<std::int64_t> checked_offsets(
    std::span<const std::int64_t> offsets, std::int64_t extent,
    const char* name) {
    if (offsets.empty()) {
        return {0, extent};
    }
    if (offsets.size() < 2 || offsets.front() != 0 || offsets.back() != extent) {
        throw std::invalid_argument(std::string(name)
                                    + " must start at zero and end at its extent");
    }
    std::int64_t previous = 0;
    for (const auto value : offsets) {
        if (value < previous || value > extent) {
            throw std::invalid_argument(std::string(name)
                                        + " must be nondecreasing and in bounds");
        }
        previous = value;
    }
    return {offsets.begin(), offsets.end()};
}

bool offsets_match(std::span<const std::int64_t> supplied,
                   const std::vector<std::int64_t>& prepared) {
    return supplied.empty()
        || (supplied.size() == prepared.size()
            && std::equal(supplied.begin(), supplied.end(), prepared.begin()));
}

std::vector<std::int64_t> repeated_bank_indices(
    const std::vector<std::int64_t>& program_offsets,
    std::int64_t num_programs) {
    std::vector<std::int64_t> indices;
    if (program_offsets.back() < 0) {
        throw std::invalid_argument("program offsets must be nonnegative");
    }
    indices.reserve(static_cast<std::size_t>(program_offsets.back()));
    for (std::size_t group = 0; group + 1 < program_offsets.size(); ++group) {
        const auto count = program_offsets[group + 1] - program_offsets[group];
        if (count != num_programs) {
            throw std::invalid_argument(
                "each default program group must contain one occurrence per bank row; "
                "supply program_bank_indices for a custom order");
        }
        for (std::int64_t index = 0; index < num_programs; ++index) {
            indices.push_back(index);
        }
    }
    return indices;
}

std::vector<std::int64_t> selected_bank_indices(
    const ProgramCall& call, const std::vector<std::int64_t>& program_offsets,
    std::int64_t target_count, std::int64_t num_programs) {
    std::vector<std::int64_t> indices;
    if (call.program_bank_indices) {
        if (call.program_bank_indices->size() != static_cast<std::size_t>(target_count)) {
            throw std::invalid_argument(
                "program_bank_indices must contain one entry per program occurrence");
        }
        indices.assign(call.program_bank_indices->begin(),
                       call.program_bank_indices->end());
        for (const auto index : indices) {
            if (index < 0 || index >= num_programs) {
                throw std::out_of_range(
                    "program_bank_indices contains an index outside the learned query bank");
            }
        }
        return indices;
    }
    return repeated_bank_indices(program_offsets, num_programs);
}

at::Tensor index_tensor(const std::vector<std::int64_t>& indices,
                        const at::Tensor& reference) {
    auto result = at::tensor(
        indices, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    return reference.device() == at::Device(at::kCPU)
        ? result : result.to(reference.device());
}

void validate_context_width_and_placement(const at::Tensor& context,
                                          const at::Tensor& features,
                                          std::int64_t context_size) {
    require_floating(context, "context");
    if (context.device() != features.device()) {
        throw std::invalid_argument("context and features must share a device");
    }
    if (context.scalar_type() != features.scalar_type()) {
        throw std::invalid_argument("context and features must share a dtype");
    }
    if (context.dim() < 1 || context.size(-1) != context_size) {
        throw std::invalid_argument("context's final dimension must equal context_size");
    }
}

at::Tensor condition_program_queries(const at::Tensor& queries,
                                     const at::Tensor& features,
                                     const at::Tensor& context,
                                     const Linear& projection,
                                     std::int64_t context_size) {
    validate_context_width_and_placement(context, features, context_size);
    auto projected = projection(context);
    if (features.dim() == 2) {
        if (context.dim() == 1) {
            return queries + projected;
        }
        if (context.dim() != 2
            || (context.size(0) != 1 && context.size(0) != queries.size(0))) {
            throw std::invalid_argument(
                "packed context must be [context_size] or "
                "[num_program_occurrences, context_size]");
        }
        return queries + projected;
    }

    const auto batch = features.size(0);
    if (context.dim() == 1) {
        return queries + projected.reshape({1, 1, queries.size(-1)});
    }
    if (context.dim() == 2) {
        if (context.size(0) != 1 && context.size(0) != batch) {
            throw std::invalid_argument(
                "dense group context must have one row per batch group");
        }
        return queries + projected.unsqueeze(1);
    }
    if (context.dim() == 3) {
        if (context.size(0) != batch || context.size(1) != queries.size(1)) {
            throw std::invalid_argument(
                "dense per-program context must be [batch, program_occurrences, context_size]");
        }
        return queries + projected;
    }
    throw std::invalid_argument("context rank must be one, two, or three");
}

void require_attention_payload(const at::Tensor& features,
                               std::int64_t width,
                               const at::Tensor& program_queries) {
    if (!features.defined() || (features.dim() != 2 && features.dim() != 3)) {
        throw std::invalid_argument(
            "features must be packed [N, width] or dense [B, N, width]");
    }
    if (features.size(-1) != width) {
        throw std::invalid_argument("features final dimension must equal width");
    }
    require_floating(features, "features");
    if (features.device() != program_queries.device()) {
        throw std::invalid_argument(
            "features and InteractionPrograms parameters must share a device");
    }
    if (features.scalar_type() != program_queries.scalar_type()) {
        throw std::invalid_argument(
            "features and InteractionPrograms parameters must share a dtype");
    }
}

void require_floating(const at::Tensor& tensor, const char* name) {
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " must be defined");
    }
    if (!tensor.is_floating_point()) {
        throw std::invalid_argument(std::string(name) + " must be floating point");
    }
}

void validate_context_groups(const at::Tensor& participants,
                             const at::Tensor& context,
                             std::int64_t context_size) {
    require_floating(context, "context");
    if (context.device() != participants.device()) {
        throw std::invalid_argument("context and participants must share a device");
    }
    if (context.scalar_type() != participants.scalar_type()) {
        throw std::invalid_argument("context and participants must share a dtype");
    }
    if (context.dim() != participants.dim() - 1 || context.size(-1) != context_size) {
        throw std::invalid_argument(
            "context must align with participant groups and end in context_size");
    }
    for (std::int64_t axis = 0; axis < participants.dim() - 2; ++axis) {
        if (context.size(axis) != participants.size(axis)) {
            throw std::invalid_argument(
                "context must align with participant groups and end in context_size");
        }
    }
}

void validate_gate_broadcast(const at::Tensor& contribution,
                             const at::Tensor& gate_logits) {
    require_floating(contribution, "gate conditioner output");
    if (contribution.device() != gate_logits.device()) {
        throw std::invalid_argument(
            "conditioner output and gate logits must share a device");
    }
    if (contribution.scalar_type() != gate_logits.scalar_type()) {
        throw std::invalid_argument(
            "conditioner output and gate logits must share a dtype");
    }

    const auto contribution_rank = contribution.dim();
    const auto logits_rank = gate_logits.dim();
    if (contribution_rank > logits_rank) {
        throw std::invalid_argument(
            "conditioner output must broadcast to gate-logit shape");
    }
    const auto shared_rank = std::min(contribution_rank, logits_rank);
    for (std::int64_t trailing = 0; trailing < shared_rank; ++trailing) {
        const auto contribution_extent = contribution.size(contribution_rank - 1 - trailing);
        const auto logit_extent = gate_logits.size(logits_rank - 1 - trailing);
        if (contribution_extent != 1 && contribution_extent != logit_extent) {
            throw std::invalid_argument(
                "conditioner output must broadcast to gate-logit shape");
        }
    }
}

} // namespace

InteractionPrograms::InteractionPrograms(
    ParameterGroup group, std::int64_t width, std::int64_t num_programs,
    std::int64_t heads, double dropout,
    std::optional<std::int64_t> context_size)
    : group_(group),
      width_(checked_program_width(width, num_programs, heads, dropout, context_size)),
      num_programs_(num_programs),
      context_size_(context_size),
      context_projection_(make_context_projection(group, context_size_, width_)),
      feature_to_program_(group, program_attention_options(width_, heads, dropout)),
      program_to_feature_(group, program_attention_options(width_, heads, dropout)) {
    program_queries_id_ = group_.normal(
        {num_programs_, width_}, std::pow(static_cast<double>(width_), -0.5));
    program_queries_ = group_.get(program_queries_id_);
}

ProgramResult InteractionPrograms::operator()(const at::Tensor& features,
                                                Run run) const {
    return apply(features, ProgramCall{}, run);
}

ProgramResult InteractionPrograms::operator()(const at::Tensor& features,
                                                const ProgramCall& call,
                                                Run run) const {
    return apply(features, call, run);
}

ProgramResult InteractionPrograms::apply(const at::Tensor& features,
                                           const ProgramCall& call,
                                           Run run) const {
    require_attention_payload(features, width_, program_queries_);
    const bool dense = features.dim() == 3;
    if (dense && call.relation != nullptr) {
        throw std::invalid_argument(
            "prepared overlapping relations require packed [N, width] features");
    }

    std::vector<std::int64_t> source_offsets;
    std::vector<std::int64_t> target_offsets;
    std::vector<std::int64_t> bank_indices;
    std::int64_t target_count = 0;
    if (call.relation != nullptr) {
        if (!call.feature_order || !call.program_order) {
            throw std::invalid_argument(
                "feature_order and program_order are required with a prepared relation");
        }
        if (!call.feature_order->valid() || !call.program_order->valid()) {
            throw std::invalid_argument(
                "prepared relation calls require valid occurrence order tokens");
        }
        const auto& relation = *call.relation;
        if (!offsets_match(call.feature_offsets, relation.source_offsets())
            || !offsets_match(call.program_offsets, relation.target_offsets())) {
            throw std::invalid_argument(
                "explicit call offsets must match the prepared relation endpoints");
        }
        target_count = relation.target_count();
        source_offsets = relation.source_offsets();
        target_offsets = relation.target_offsets();
        if (call.structural_generation) {
            relation.validate_for(*call.feature_order, *call.program_order,
                                  features.size(0), target_count,
                                  *call.structural_generation, features.device());
        } else {
            relation.validate_for(*call.feature_order, *call.program_order,
                                  features.size(0), target_count, features.device());
        }
        bank_indices = selected_bank_indices(
            call, target_offsets, target_count, num_programs_);
    } else if (dense) {
        if (!call.feature_offsets.empty() || !call.program_offsets.empty()) {
            throw std::invalid_argument(
                "dense attention uses the batch dimension for grouping; offsets are packed-only");
        }
        if (call.program_bank_indices) {
            target_count = static_cast<std::int64_t>(call.program_bank_indices->size());
        } else {
            target_count = num_programs_;
        }
        target_offsets = {0, target_count};
        bank_indices = selected_bank_indices(
            call, target_offsets, target_count, num_programs_);
    } else {
        source_offsets = checked_offsets(
            call.feature_offsets, features.size(0), "feature_offsets");
        const auto num_groups = static_cast<std::int64_t>(source_offsets.size() - 1);
        if (call.program_offsets.empty()) {
            if (call.program_bank_indices && num_groups != 1) {
                throw std::invalid_argument(
                    "program_offsets are required with explicit bank indices and multiple groups");
            }
            if (call.program_bank_indices) {
                target_count = static_cast<std::int64_t>(
                    call.program_bank_indices->size());
                target_offsets = {0, target_count};
            } else {
                target_offsets.reserve(static_cast<std::size_t>(num_groups + 1));
                target_offsets.push_back(0);
                for (std::int64_t group_index = 0; group_index < num_groups;
                     ++group_index) {
                    target_offsets.push_back(target_offsets.back() + num_programs_);
                }
                target_count = target_offsets.back();
            }
        } else {
            if (call.program_offsets.size() < 2
                || call.program_offsets.front() != 0) {
                throw std::invalid_argument(
                    "program_offsets must start at zero and include a terminal offset");
            }
            target_count = call.program_offsets.back();
            target_offsets = checked_offsets(call.program_offsets, target_count,
                                              "program_offsets");
        }
        if (target_offsets.size() != source_offsets.size()) {
            throw std::invalid_argument(
                "feature_offsets and program_offsets must describe the same groups");
        }
        bank_indices = selected_bank_indices(
            call, target_offsets, target_count, num_programs_);
    }

    const auto indices = index_tensor(bank_indices, program_queries_);
    auto queries = program_queries_.index_select(0, indices);
    if (dense) {
        queries = queries.unsqueeze(0).expand(
            {features.size(0), target_count, width_});
    }
    if (call.context.defined()) {
        if (!context_projection_) {
            throw std::invalid_argument(
                "context was supplied but InteractionPrograms has no context_size");
        }
        queries = condition_program_queries(queries, features, call.context,
                                            *context_projection_, *context_size_);
    }

    at::Tensor program_features;
    at::Tensor feature_updates;
    if (call.relation != nullptr) {
        program_features = feature_to_program_.cross(
            features, queries, *call.relation, run);
        const auto reverse = call.relation->transpose();
        feature_updates = program_to_feature_.cross(
            program_features, features, reverse, run);
    } else if (dense) {
        program_features = feature_to_program_.cross(
            features, queries, AttentionLayout{}, run);
        feature_updates = program_to_feature_.cross(
            program_features, features, AttentionLayout{}, run);
    } else {
        const auto target_view = std::span<const std::int64_t>(target_offsets);
        const AttentionLayout forward_layout{source_offsets, target_view, false};
        const AttentionLayout reverse_layout{target_view, source_offsets, false};
        program_features = feature_to_program_.cross(
            features, queries, forward_layout, run);
        feature_updates = program_to_feature_.cross(
            program_features, features, reverse_layout, run);
    }
    return {feature_updates, program_features};
}

GatedInteractor::GatedInteractor(ParameterGroup group, std::int64_t input_size,
                                 std::int64_t hidden_size,
                                 std::int64_t output_size, bool normalize,
                                 std::optional<std::int64_t> context_size)
    : group_(group),
      input_size_(checked_positive(input_size, "input_size")),
      output_size_(checked_positive(output_size, "output_size")),
      normalize_(normalize),
      context_size_(checked_context_size(context_size)),
      encode_(group, input_size_, checked_positive(hidden_size, "hidden_size")),
      gate_(group, input_size_, 1),
      context_projection_(make_context_projection(group, context_size_, input_size_)),
      joint_hidden_(group, checked_positive(hidden_size, "hidden_size"), hidden_size),
      joint_output_(group, hidden_size, output_size_) {}

at::Tensor GatedInteractor::operator()(const at::Tensor& participants, Run run) const {
    return apply(participants, {}, {}, run);
}

at::Tensor GatedInteractor::operator()(const at::Tensor& participants,
                                       const at::Tensor& context,
                                       Run run) const {
    if (!context_projection_) {
        throw std::invalid_argument(
            "context was supplied but GatedInteractor has no context_size");
    }
    return apply(participants, context, {}, run);
}

at::Tensor GatedInteractor::apply_gate_conditioner(
    const at::Tensor& participants, const at::Tensor& gate_logit_addition,
    Run run) const {
    if (context_projection_) {
        throw std::invalid_argument(
            "context_size and an explicit gate conditioner are mutually exclusive");
    }
    if (!gate_logit_addition.defined()) {
        throw std::invalid_argument("conditioner must return a defined tensor");
    }
    return apply(participants, {}, gate_logit_addition, run);
}

at::Tensor GatedInteractor::apply(const at::Tensor& participants,
                                  const at::Tensor& context,
                                  const at::Tensor& gate_logit_addition,
                                  Run run) const {
    static_cast<void>(run);
    require_floating(participants, "participants");
    if (participants.dim() < 2) {
        throw std::invalid_argument(
            "participants need [..., participants, features] dimensions");
    }
    if (participants.size(-1) != input_size_) {
        throw std::invalid_argument(
            "participant feature size does not match input_size");
    }
    if (participants.device() != encode_.weight().device()) {
        throw std::invalid_argument(
            "participants and GatedInteractor parameters must share a device");
    }
    if (participants.scalar_type() != encode_.weight().scalar_type()) {
        throw std::invalid_argument(
            "participants and GatedInteractor parameters must share a dtype");
    }

    at::Tensor conditioned = participants;
    if (context.defined()) {
        if (!context_projection_) {
            throw std::invalid_argument(
                "context was supplied but GatedInteractor has no context_size");
        }
        validate_context_groups(participants, context, *context_size_);
        conditioned = participants + (*context_projection_)(context, run).unsqueeze(-2);
    }

    auto hidden = at::tanh(encode_(conditioned, run));
    auto gate_logits = gate_(conditioned, run);
    if (gate_logit_addition.defined()) {
        validate_gate_broadcast(gate_logit_addition, gate_logits);
        gate_logits = gate_logits + gate_logit_addition;
    }
    auto weights = at::sigmoid(gate_logits);
    auto pooled = (weights * hidden).sum(-2);
    if (normalize_) {
        pooled = pooled / weights.sum(-2).clamp_min(1.0);
    }
    pooled = at::silu(pooled);
    pooled = at::silu(joint_hidden_(pooled, run));
    return joint_output_(pooled, run);
}

} // namespace cellerator::nn
