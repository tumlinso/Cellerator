#include <Cellerator/compiler/planning/adapt_multi_extent_direct_binding_and_assembly_fallback_v1.hh>

#include <cassert>
#include <cstdint>

namespace planning = Cellerator::compiler::planning;
namespace joint = cellerator::execution::joint_compiler;
namespace execution = cellerator::execution;

int main() {
    alignas(16) std::uint8_t first[32]{};
    alignas(16) std::uint8_t second[48]{};
    const joint::opaque_runtime_token_v1 readiness{5u, 6u};
    const joint::opaque_runtime_token_v1 lease{7u, 8u};
    joint::external_extent_v1 extents[2]{};
    extents[0].address = first;
    extents[0].location = {execution::residency_kind::host, {}, -1, 1u};
    extents[0].bytes = sizeof(first);
    extents[0].alignment = 16u;
    extents[0].order = {1u, 2u};
    extents[0].generation = {3u};
    extents[0].readiness = readiness;
    extents[0].lease = lease;
    extents[1] = extents[0];
    extents[1].address = second;
    extents[1].plane_byte_offset = sizeof(first);
    extents[1].bytes = sizeof(second);
    extents[1].readiness = {5u, 7u};
    extents[1].lease = {7u, 9u};

    joint::external_binding_v1 binding{};
    binding.binding_identity = {9u, 10u};
    binding.atom_identity = {11u, 12u};
    binding.plane_identity = {13u, 14u};
    binding.extents = extents;
    binding.extent_count = 2u;
    binding.total_bytes = sizeof(first) + sizeof(second);

    const planning::multi_extent_candidate_capability_v1 direct_capability{
        21u, true, 4u};
    const auto direct = planning::plan_multi_extent_binding_v1(
        binding, direct_capability, 100u, 25u, 16u);
    assert(direct);
    assert(direct.plan.route == planning::multi_extent_binding_route_v1::direct);
    assert(direct.plan.assembly.bytes_copied == 0u);
    assert(!direct.plan.assembly_profiler_stage_visible);
    assert(direct.plan.total_predicted_nanoseconds == 100u);

    const planning::multi_extent_candidate_capability_v1 assembly_capability{
        22u, false, 0u};
    const auto assembled = planning::plan_multi_extent_binding_v1(
        binding, assembly_capability, 100u, 25u, 16u);
    assert(assembled);
    assert(assembled.plan.route == planning::multi_extent_binding_route_v1::assembled);
    assert(assembled.plan.assembly.bytes_copied == 80u);
    assert(assembled.plan.assembly.copy_operations == 2u);
    assert(assembled.plan.assembly.predicted_nanoseconds == 30u);
    assert(assembled.plan.assembly_profiler_stage_visible);
    assert(assembled.plan.total_predicted_nanoseconds == 130u);

    const auto invalid_cost = planning::plan_multi_extent_binding_v1(
        binding, assembly_capability, 100u, 25u, 0u);
    assert(!invalid_cost);
    assert(invalid_cost.code == planning::multi_extent_binding_plan_code_v1::invalid_cost_model);
}
