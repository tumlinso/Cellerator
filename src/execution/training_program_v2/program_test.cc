#include <Cellerator/execution/training_program_v2/program.hh>

#include <array>
#include <cstdint>

namespace {

using namespace cellerator::execution;
using namespace cellerator::execution::training_v2;

axis_identity axis(std::uint32_t value) {
    return {{value, 1u}, {value + 1u, 1u}, {value + 2u, 1u},
        {value + 3u, 1u}};
}

training_stage_v2 stage(training_stage_kind_v2 kind,
    std::uint64_t identity) {
    training_stage_v2 result{};
    result.kind = kind;
    result.stage_identity = identity;
    result.candidate_identity = identity + 100u;
    result.input_axis = axis(1u);
    result.output_axis = axis(10u);
    return result;
}

training_program_v2 program(const training_stage_v2 *stages,
    std::uint32_t count) {
    training_program_v2 result{};
    result.stage_count = count;
    result.stages = stages;
    result.program_identity = 20u;
    result.structure = {30u, 1u};
    result.epoch = {40u};
    result.prepared_generation = {50u};
    result.source_axis = axis(1u);
    result.destination_axis = axis(10u);
    result.value_mode = training_value_mode_v2::projection_primary;
    result.internal_order = training_order_mode_v2::persistent_physical;
    result.graph_capture_required = true;
    return result;
}

} // namespace

int main() {
    using namespace cellerator::execution::training_v2;
    const std::array<training_stage_v2, 3> stages{{
        stage(training_stage_kind_v2::forward_relation_apply, 1u),
        stage(training_stage_kind_v2::transpose_relation_apply, 2u),
        stage(training_stage_kind_v2::logical_edge_gradient, 3u)}};
    training_program_receipt_v2 receipt{};
    if (!validate_training_program_v2(
            program(stages.data(), stages.size()), receipt))
        return 1;
    if (!receipt.has_forward || !receipt.has_transpose
        || !receipt.has_edge_gradient || receipt.production_promoted)
        return 2;

    auto missing = stages;
    missing[1].kind = training_stage_kind_v2::forward_relation_apply;
    if (validate_training_program_v2(
            program(missing.data(), missing.size()), receipt))
        return 3;
    auto promoted = stages;
    promoted[2].production_promoted = true;
    if (validate_training_program_v2(
            program(promoted.data(), promoted.size()), receipt))
        return 4;
    return 0;
}
