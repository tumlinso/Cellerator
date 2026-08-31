#include <Cellerator/execution/training_program_v2/program.hh>

#include <cstdint>
#include <limits>

namespace cellerator::execution::training_v2 {
namespace {

training_result_v2 error(training_status_v2 code, const char *message) noexcept {
    return {code, message};
}

bool valid_stage_kind(training_stage_kind_v2 value) noexcept {
    return value == training_stage_kind_v2::forward_relation_apply
        || value == training_stage_kind_v2::transpose_relation_apply
        || value == training_stage_kind_v2::logical_edge_gradient
        || value == training_stage_kind_v2::sparse_axis_update
        || value == training_stage_kind_v2::publish_value_generation
        || value == training_stage_kind_v2::explicit_canonicalize;
}

bool valid_order(training_order_mode_v2 value) noexcept {
    return value == training_order_mode_v2::canonical
        || value == training_order_mode_v2::persistent_physical;
}

bool valid_value_mode(training_value_mode_v2 value) noexcept {
    return value == training_value_mode_v2::logical_primary
        || value == training_value_mode_v2::projection_primary;
}

bool add_fits(std::uint64_t &total, std::uint64_t value) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - total) return false;
    total += value;
    return true;
}

} // namespace

training_result_v2 validate_training_program_v2(
    const training_program_v2 &program,
    training_program_receipt_v2 &receipt) noexcept {
    receipt = {};
    if (program.schema_version != training_program_schema_version_v2)
        return error(training_status_v2::unsupported_schema,
            "training program v2 schema is unsupported");
    if (program.stage_count < 3u || program.stages == nullptr
        || !valid_value_mode(program.value_mode)
        || !valid_order(program.internal_order)
        || program.program_identity == 0u
        || !valid_handle(program.structure)
        || program.epoch.value == 0u
        || program.prepared_generation.value == 0u
        || !valid_axis_identity(program.source_axis)
        || !valid_axis_identity(program.destination_axis))
        return error(training_status_v2::invalid_identity,
            "training program identity or graph envelope is invalid");
    if (program.numerical.input_type != numeric_type::f32
        || program.numerical.accumulation_type != numeric_type::f32
        || program.numerical.output_type != numeric_type::f32
        || (program.numerical.nonfinite
                != training_nonfinite_policy_v2::propagate
            && program.numerical.nonfinite
                != training_nonfinite_policy_v2::reject)
        || !program.numerical.deterministic)
        return error(training_status_v2::unsupported_numeric_policy,
            "training program requires explicit deterministic FP32 policy");

    bool update_seen = false;
    bool publication_seen = false;
    bool canonicalization_seen = false;
    bool graph_compatible = true;
    std::uint64_t persistent = 0u;
    std::uint64_t transient = 0u;
    std::uint64_t launch_count = 0u;
    for (std::uint32_t index = 0u; index < program.stage_count; ++index) {
        const training_stage_v2 &stage = program.stages[index];
        if (!valid_stage_kind(stage.kind) || !valid_order(stage.input_order)
            || !valid_order(stage.output_order)
            || stage.reserved0 != 0u || stage.reserved1 != 0u
            || stage.stage_identity == 0u || stage.candidate_identity == 0u
            || !valid_axis_identity(stage.input_axis)
            || !valid_axis_identity(stage.output_axis)
            || stage.launch_count == 0u || !stage.requires_measurement
            || stage.production_promoted)
            return error(training_status_v2::invalid_stage_graph,
                "training stage contract is invalid or promoted");
        for (std::uint8_t value : stage.reserved2)
            if (value != 0u)
                return error(training_status_v2::invalid_argument,
                    "training stage reserved byte is nonzero");
        if (publication_seen)
            return error(training_status_v2::invalid_stage_graph,
                "value publication must be the final stage");
        receipt.has_forward |=
            stage.kind == training_stage_kind_v2::forward_relation_apply;
        receipt.has_transpose |=
            stage.kind == training_stage_kind_v2::transpose_relation_apply;
        receipt.has_edge_gradient |=
            stage.kind == training_stage_kind_v2::logical_edge_gradient;
        update_seen |= stage.kind == training_stage_kind_v2::sparse_axis_update;
        publication_seen |=
            stage.kind == training_stage_kind_v2::publish_value_generation;
        canonicalization_seen |=
            stage.kind == training_stage_kind_v2::explicit_canonicalize;
        graph_compatible &= stage.graph_capture_compatible;
        if (!add_fits(persistent, stage.persistent_bytes)
            || !add_fits(transient, stage.transient_bytes)
            || !add_fits(launch_count, stage.launch_count))
            return error(training_status_v2::invalid_stage_graph,
                "training program resource census overflows");
    }
    if (!receipt.has_forward || !receipt.has_transpose
        || !receipt.has_edge_gradient || (update_seen != publication_seen))
        return error(training_status_v2::invalid_stage_graph,
            "training graph lacks required forward/backward/gradient stages");
    if (program.internal_order == training_order_mode_v2::persistent_physical
        && program.canonical_output_required && !canonicalization_seen)
        return error(training_status_v2::invalid_stage_graph,
            "canonical output requires an explicit canonicalization stage");
    if (program.graph_capture_required && !graph_compatible)
        return error(training_status_v2::invalid_stage_graph,
            "training graph capture requirement is not satisfied");
    if (launch_count > std::numeric_limits<std::uint32_t>::max())
        return error(training_status_v2::invalid_stage_graph,
            "training program launch count exceeds receipt capacity");
    receipt.validated_stage_count = program.stage_count;
    receipt.launch_count = static_cast<std::uint32_t>(launch_count);
    receipt.persistent_bytes = persistent;
    receipt.transient_bytes = transient;
    receipt.graph_capture_compatible = graph_compatible;
    receipt.requires_measurement = true;
    receipt.production_promoted = false;
    return {};
}

} // namespace cellerator::execution::training_v2
