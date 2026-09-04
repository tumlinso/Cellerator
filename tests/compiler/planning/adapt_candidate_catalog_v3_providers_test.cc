#include <Cellerator/compiler/planning/adapt_candidate_catalog_v3_providers_v1.hh>

#include <cassert>
#include <cstring>

namespace planning = Cellerator::compiler::planning;
namespace catalog = cellerator::compute::operation::catalog_v3;

namespace {

bool prepare(std::uint64_t candidate, const void* context) noexcept {
    return candidate == *static_cast<const std::uint64_t*>(context);
}

template <std::size_t N>
void set_name(char (&output)[N], const char* value) {
    std::strncpy(output, value, N - 1u);
}

}  // namespace

int main() {
    const auto inventory = catalog::built_in_provider_operation_inventory_v3();

    catalog::candidate_stage_v3 stage{};
    stage.stage_id = 30u;
    stage.kernel_id = 31u;
    stage.stage_kind = 1u;
    stage.launch_count = 2u;
    set_name(stage.stable_name, "apply");
    catalog::candidate_descriptor_v3 descriptor{};
    descriptor.identity.candidate_id = 40u;
    descriptor.identity.provider_id = inventory.providers[0].stable_provider_id;
    descriptor.identity.device_class_id = 41u;
    descriptor.identity.projection_type_id = 42u;
    descriptor.identity.capability_id = 43u;
    descriptor.identity.operation_id = inventory.operations[0].stable_operation_id;
    descriptor.identity.width_min = 1u;
    descriptor.identity.width_max = 16u;
    descriptor.stages = &stage;
    descriptor.stage_count = 1u;
    descriptor.resources.threads_per_cta = 128u;
    const catalog::candidate_catalog_view_v3 source{&descriptor, 1u};
    const std::uint64_t expected_candidate = 40u;
    const planning::source_linked_preparation_hook_v1 hook{
        40u, 0u, prepare, &expected_candidate};

    const auto adapted = planning::adapt_candidate_catalog_v3_to_planning_ir_v1(
        source, inventory, &hook, 1u);
    assert(adapted);
    assert(adapted.ir.providers.size() == inventory.provider_count);
    assert(adapted.ir.operations.size() == inventory.operation_count);
    assert(adapted.ir.candidates.size() == 1u);
    assert(adapted.ir.candidates[0].stages[0].stable_name == "apply");
    assert(adapted.ir.candidates[0].resources.threads_per_cta == 128u);
    assert(adapted.ir.candidates[0].preparation.prepare(
        adapted.ir.candidates[0].candidate_identity,
        adapted.ir.candidates[0].preparation.context));
    assert(planning::cross_validate_candidate_catalog_planning_ir_v1(
        source, inventory, adapted.ir));

    auto missing_provider = descriptor;
    missing_provider.identity.provider_id = 99u;
    const catalog::candidate_catalog_view_v3 invalid_source{&missing_provider, 1u};
    const auto rejected = planning::adapt_candidate_catalog_v3_to_planning_ir_v1(
        invalid_source, inventory, &hook, 1u);
    assert(!rejected);
    assert(rejected.code == planning::candidate_catalog_adapter_code_v1::missing_provider);
}
