#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/contract_catalog_v1.hh>

namespace cellerator::compute::architecture::providers::nvidia::sm70::contract {
namespace {

constexpr std::uint64_t fnv1a(const char *text,
    std::uint64_t value = 1469598103934665603ull) noexcept {
    return *text == '\0' ? value
        : fnv1a(text + 1,
            (value ^ static_cast<std::uint8_t>(*text)) * 1099511628211ull);
}

constexpr catalog_entry_v1 entries[] = {
    {fnv1a("sm70.contract.sparse.thread_per_edge.v1"),
        "sm70.contract.sparse.thread_per_edge.v1",
        candidate_kind_v1::sparse_thread_per_edge, 1u, 32u, 1u, 1u,
        true, true, true, false, true, true, false},
    {fnv1a("sm70.contract.sparse.warp_per_edge.v1"),
        "sm70.contract.sparse.warp_per_edge.v1",
        candidate_kind_v1::sparse_warp_per_edge, 17u, 256u, 1u, 1u,
        true, true, true, false, true, true, false},
    {fnv1a("sm70.contract.sparse.cooperative_group.v1"),
        "sm70.contract.sparse.cooperative_group.v1",
        candidate_kind_v1::sparse_cooperative_group, 65u, 0xffffffffu,
        1u, 1u, true, true, true, false, true, true, false},
    {fnv1a("sm70.contract.rectangular_mma_exact_residual.v1"),
        "sm70.contract.rectangular_mma_exact_residual.v1",
        candidate_kind_v1::rectangular_mma_exact_residual, 16u,
        0xffffffffu, 16u, 16u, true, false, true, true, true, true,
        false},
};

constexpr std::size_t entry_count = sizeof(entries) / sizeof(entries[0]);

} // namespace

const catalog_entry_v1 *catalog_v1(std::size_t *count) noexcept {
    if (count != nullptr) *count = entry_count;
    return entries;
}

planner_candidate_v1 evaluate_candidate_v1(const planner_problem_v1 &problem,
    candidate_kind_v1 candidate) noexcept {
    planner_candidate_v1 result{};
    const std::size_t index = static_cast<std::size_t>(candidate);
    if (index >= entry_count) return result;
    result.entry = &entries[index];
    const catalog_entry_v1 &entry = *result.entry;
    const bool output_supported =
        problem.required_output_order == output_order_v1::logical_edge
            ? entry.supports_logical_output
            : entry.supports_projection_output;
    const bool cover_available = !entry.requires_rectangular_cover
        || problem.rectangular_tile_count != 0u;
    result.eligible = problem.local_edge_count != 0u
        && problem.dense_width >= entry.minimum_width
        && problem.dense_width <= entry.maximum_width && output_supported
        && cover_available && problem.cuda_execution_resource_available;
    result.empirical_measurement_required = entry.requires_measurement;
    return result;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::contract
