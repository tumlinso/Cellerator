#include <Cellerator/compiler/ir/planning/implement_candidate_family_and_provider_nodes_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::ir::planning::v1;
    catalog_v3::candidate_stage_v3 stages[2]{};
    stages[0].stage_id = 1u;
    stages[1].stage_id = 2u;
    catalog_v3::candidate_descriptor_v3 source{};
    source.identity.candidate_id = 10u;
    source.identity.provider_id = 11u;
    source.identity.device_class_id = 12u;
    source.identity.projection_type_id = 13u;
    source.identity.capability_id = 14u;
    source.identity.operation_id = 15u;
    source.identity.width_min = 2u;
    source.identity.width_max = 64u;
    source.identity.numerics = catalog_v3::numerical_mode::relaxed;
    source.identity.classification = catalog_v3::candidate_class::experimental;
    source.identity.requires_measurement = true;
    source.resources = {1024u, 2048u, 128u, 4096u};
    source.stages = stages;
    source.stage_count = 2u;

    candidate_provider_node_v1 imported{};
    assert(import_candidate_catalog_v3(source, {20u, 21u}, {30u, 31u}, 40u,
                                       &imported) == candidate_provider_status_v1::ok);
    assert(imported.candidate_id == source.identity.candidate_id);
    assert(imported.provider.low == source.identity.provider_id);
    assert(imported.device_class_id == source.identity.device_class_id);
    assert(imported.projection_type_id == source.identity.projection_type_id);
    assert(imported.capability_id == source.identity.capability_id);
    assert(imported.operation_id == source.identity.operation_id);
    assert(imported.width_min == source.identity.width_min);
    assert(imported.width_max == source.identity.width_max);
    assert(imported.numerics == source.identity.numerics);
    assert(imported.resources.transient_bytes == source.resources.transient_bytes);
    assert(imported.stages == source.stages && imported.stage_count == source.stage_count);
    assert((imported.flags & candidate_provider_experimental_v1) != 0u);
    assert((imported.flags & candidate_provider_requires_measurement_v1) != 0u);
    assert((imported.flags & candidate_provider_source_extension_v1) != 0u);
    assert(imported.preparation_entrypoint == 40u);

    source.identity.width_max = 1u;
    assert(import_candidate_catalog_v3(source, {20u, 21u}, {}, 0u, &imported) ==
           candidate_provider_status_v1::invalid_width);
}
