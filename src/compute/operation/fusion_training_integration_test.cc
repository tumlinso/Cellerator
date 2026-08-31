#include <Cellerator/compute/compute.hh>

#include <array>
#include <cassert>
#include <cstdint>

using namespace cellerator::execution;
using namespace cellerator::execution::training_v2;

namespace {

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

} // namespace

int main() {
    namespace fusion = cellerator::compute::operation::fusion;
    std::size_t fusion_count = 0u;
    const fusion::registry_entry_v1 *entries =
        fusion::fusion_registry_v1(&fusion_count);
    assert(entries != nullptr && fusion_count == 20u);
    assert(fusion::validate_fusion_registry_v1() == fusion::status_v1::success);
    for (std::size_t left = 0u; left < fusion_count; ++left) {
        assert(entries[left].stable_candidate_id != 0u
            && entries[left].requires_measurement
            && !entries[left].auto_promoted
            && entries[left].unfused_stages_available);
        for (std::size_t right = left + 1u; right < fusion_count; ++right)
            assert(entries[left].stable_candidate_id
                != entries[right].stable_candidate_id);
    }
    std::array<fusion::stage_descriptor_v1, 2> fusion_stages{};
    fusion_stages[0].stable_stage_id = 1001u;
    fusion_stages[0].kind = fusion::stage_kind_v1::value_pack;
    fusion_stages[0].output_order = fusion::order_kind_v1::projection_native;
    fusion_stages[0].structure_id = 2001u;
    fusion_stages[0].structure_epoch = 3u;
    fusion_stages[0].input_value_generation = 4u;
    fusion_stages[0].output_value_generation = 5u;
    fusion_stages[0].global_item_begin = std::uint64_t{1} << 32u;
    fusion_stages[0].local_item_count = 7u;
    fusion_stages[0].profiler_stage_index = 11u;
    fusion_stages[1] = fusion_stages[0];
    fusion_stages[1].stable_stage_id = 1002u;
    fusion_stages[1].kind = fusion::stage_kind_v1::relation_apply;
    fusion_stages[1].input_order = fusion::order_kind_v1::projection_native;
    fusion_stages[1].input_value_generation = 5u;
    fusion_stages[1].profiler_stage_index = 12u;
    const fusion::dependency_v1 dependency{0u, 1u};
    fusion::prepared_stage_graph_v1 graph{};
    graph.stable_graph_id = 3001u;
    graph.composition = fusion::composition_kind_v1::value_pack_to_relation_apply;
    graph.fused = true;
    graph.stages = fusion_stages.data();
    graph.stage_count = static_cast<std::uint32_t>(fusion_stages.size());
    graph.dependencies = &dependency;
    graph.dependency_count = 1u;
    assert(fusion::validate_prepared_stage_graph_v1(graph)
        == fusion::status_v1::success);
    fusion_stages[1].stable_stage_id = fusion_stages[0].stable_stage_id;
    assert(fusion::validate_prepared_stage_graph_v1(graph)
        == fusion::status_v1::invalid_identity);
    fusion_stages[1].stable_stage_id = 1002u;

    const std::array<training_stage_v2, 3> stages{{
        stage(training_stage_kind_v2::forward_relation_apply, 1u),
        stage(training_stage_kind_v2::transpose_relation_apply, 2u),
        stage(training_stage_kind_v2::logical_edge_gradient, 3u)}};
    training_program_v2 program{};
    program.stage_count = static_cast<std::uint32_t>(stages.size());
    program.stages = stages.data();
    program.program_identity = 20u;
    program.structure = {30u, 1u};
    program.epoch = {40u};
    program.prepared_generation = {50u};
    program.source_axis = axis(1u);
    program.destination_axis = axis(10u);
    program.value_mode = training_value_mode_v2::projection_primary;
    program.internal_order = training_order_mode_v2::persistent_physical;
    program.graph_capture_required = true;
    training_program_receipt_v2 receipt{};
    assert(validate_training_program_v2(program, receipt));
    assert(receipt.has_forward && receipt.has_transpose
        && receipt.has_edge_gradient && !receipt.production_promoted);
}
