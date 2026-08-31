#include <Cellerator/compute/operation/fusion/prepared_stage_graph_v1.hh>

#include <limits>

namespace cellerator::compute::operation::fusion {
namespace {

bool valid_stage_kind(stage_kind_v1 kind) noexcept {
    return kind <= stage_kind_v1::relation_moments;
}

bool valid_order(order_kind_v1 order) noexcept {
    return order <= order_kind_v1::persistent_physical;
}

} // namespace

status_v1 validate_prepared_stage_graph_v1(
    const prepared_stage_graph_v1 &graph) noexcept {
    if (graph.stable_graph_id == 0u || graph.stages == nullptr
        || graph.stage_count < 2u || graph.dependencies == nullptr
        || graph.dependency_count == 0u)
        return status_v1::invalid_argument;
    if (graph.composition > composition_kind_v1::relation_moments_pair)
        return status_v1::unsupported;
    if (!graph.experimental || !graph.requires_measurement
        || !graph.explicitly_selectable || graph.auto_promoted
        || !graph.unfused_stages_available || graph.reserved[0] != 0u
        || graph.reserved[1] != 0u)
        return status_v1::invalid_argument;
    for (std::uint32_t index = 0u; index < graph.stage_count; ++index) {
        const stage_descriptor_v1 &stage = graph.stages[index];
        if (stage.stable_stage_id == 0u || !valid_stage_kind(stage.kind)
            || !valid_order(stage.input_order) || !valid_order(stage.output_order)
            || stage.reserved != 0u || stage.structure_id == 0u
            || stage.structure_epoch == 0u
            || stage.input_value_generation == 0u
            || stage.output_value_generation == 0u
            || stage.local_item_count == 0u
            || stage.global_item_begin
                > std::numeric_limits<std::uint64_t>::max()
                    - stage.local_item_count)
            return status_v1::invalid_identity;
        for (std::uint32_t prior = 0u; prior < index; ++prior)
            if (graph.stages[prior].stable_stage_id == stage.stable_stage_id
                || graph.stages[prior].profiler_stage_index
                    == stage.profiler_stage_index)
                return status_v1::invalid_identity;
    }
    for (std::uint32_t index = 0u; index < graph.dependency_count; ++index) {
        const dependency_v1 dependency = graph.dependencies[index];
        if (dependency.producer_stage >= graph.stage_count
            || dependency.consumer_stage >= graph.stage_count
            || dependency.producer_stage >= dependency.consumer_stage)
            return status_v1::invalid_dependency;
        const stage_descriptor_v1 &producer =
            graph.stages[dependency.producer_stage];
        const stage_descriptor_v1 &consumer =
            graph.stages[dependency.consumer_stage];
        if (producer.output_order != consumer.input_order)
            return status_v1::incompatible_order;
        if (producer.structure_id != consumer.structure_id
            || producer.structure_epoch != consumer.structure_epoch
            || producer.output_value_generation
                != consumer.input_value_generation)
            return status_v1::incompatible_lifetime;
    }
    return status_v1::success;
}

status_v1 validate_graph_resources_v1(const prepared_stage_graph_v1 &graph,
    resource_availability_v1 resources) noexcept {
    const status_v1 graph_status = validate_prepared_stage_graph_v1(graph);
    if (graph_status != status_v1::success) return graph_status;
    std::uint32_t required = resource_none_v1;
    for (std::uint32_t index = 0u; index < graph.stage_count; ++index)
        required |= graph.stages[index].required_resources;
    if ((required & ~resources.available_flags) != 0u)
        return status_v1::invalid_argument;
    if ((required & resource_transient_workspace_v1) != 0u
        && resources.transient_workspace_bytes == 0u)
        return status_v1::invalid_argument;
    return status_v1::success;
}

} // namespace cellerator::compute::operation::fusion
