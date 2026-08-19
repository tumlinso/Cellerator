#include <Cellerator/compute/math/physical_bell.hh>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::compute::math {
namespace physical_bell_detail {

bell_lowering_status validate(const bell_csr_source_view &,
    const bell_semantic_plan_view &, const cellpack::local_cell_order_view &,
    const bell_lowering_workspace &, std::uint64_t *) noexcept;

bell_lowering_status validate_bell_shapes(
    const bell_csr_source_view &source,
    const bell_semantic_plan_view &plan,
    const cellpack::local_cell_order_view &order) noexcept {
    if (plan.semantic_schema_version != cellpack::packing_plan_semantic_schema_version
        || plan.full_row_count == 0u || plan.feature_count == 0u
        || plan.feature_block_count == 0u
        || plan.feature_block_geometry_identity == 0u || plan.row_domain_identity == 0u
        || plan.feature_block_offsets == nullptr || plan.feature_permutation == nullptr
        || plan.inverse_feature_permutation == nullptr)
        return {bell_lowering_status_code::incompatible_plan, "semantic plan is incomplete"};
    if (source.row_count != order.row_count || source.feature_count != plan.feature_count
        || source.value_size_bytes == 0u || source.value_size_bytes > 16u
        || source.row_offsets == nullptr
        || (source.nnz_count != 0u && (source.feature_ids == nullptr || source.values == nullptr)))
        return {bell_lowering_status_code::invalid_source, "canonical CSR is invalid"};
    if (order.order_schema_version != cellpack::local_cell_order_schema_version
        || order.signature_algorithm_version != cellpack::local_cell_signature_algorithm_version
        || order.ordering_identity == 0u || order.window_size == 0u || order.group_width == 0u
        || order.full_row_count != plan.full_row_count
        || order.feature_block_count != plan.feature_block_count
        || order.feature_block_geometry_identity != plan.feature_block_geometry_identity
        || order.row_domain_identity != plan.row_domain_identity
        || order.row_permutation == nullptr || order.inverse_row_permutation == nullptr
        || order.global_row_begin > order.full_row_count
        || order.row_count > order.full_row_count - order.global_row_begin)
        return {bell_lowering_status_code::incompatible_order, "row order and plan disagree"};
    if (plan.feature_block_offsets[0] != 0u
        || plan.feature_block_offsets[plan.feature_block_count] != plan.feature_count)
        return {bell_lowering_status_code::incompatible_plan, "feature-block endpoints are invalid"};
    for (std::uint32_t block = 0u; block < plan.feature_block_count; ++block)
        if (plan.feature_block_offsets[block] >= plan.feature_block_offsets[block + 1u])
            return {bell_lowering_status_code::incompatible_plan, "feature block is empty"};
    return {};
}

namespace {

constexpr std::uint64_t fnv_offset = 1469598103934665603ull;
constexpr std::uint64_t fnv_prime = 1099511628211ull;

void hash(std::uint64_t *value, std::uint64_t item) noexcept {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        *value ^= (item >> (byte * 8u)) & 0xffu;
        *value *= fnv_prime;
    }
}

std::uint32_t semantic_block(
    const bell_semantic_plan_view &plan,
    std::uint32_t execution_feature) noexcept {
    std::uint32_t first = 0u, last = plan.feature_block_count;
    while (first + 1u < last) {
        const std::uint32_t middle = first + (last - first) / 2u;
        if (plan.feature_block_offsets[middle] <= execution_feature) first = middle;
        else last = middle;
    }
    return first;
}

bool multiply(std::uint64_t left, std::uint64_t right, std::uint64_t *out) noexcept {
    if (left != 0u && right > std::numeric_limits<std::uint64_t>::max() / left)
        return false;
    *out = left * right;
    return true;
}

} // namespace

std::uint32_t physical_block(
    const bell_semantic_plan_view &plan,
    const std::uint32_t *offsets,
    std::uint32_t execution,
    std::uint32_t size) noexcept {
    const std::uint32_t semantic = semantic_block(plan, execution);
    return offsets[semantic]
        + (execution - plan.feature_block_offsets[semantic]) / size;
}

bool fill_physical_offsets(
    const bell_semantic_plan_view &plan,
    std::uint32_t size,
    std::uint32_t *offsets,
    std::uint32_t *physical_blocks) noexcept {
    std::uint64_t count = 0u;
    offsets[0] = 0u;
    for (std::uint32_t block = 0u; block < plan.feature_block_count; ++block) {
        const std::uint64_t width = plan.feature_block_offsets[block + 1u]
            - plan.feature_block_offsets[block];
        count += (width + size - 1u) / size;
        if (count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
            return false;
        offsets[block + 1u] = static_cast<std::uint32_t>(count);
    }
    *physical_blocks = static_cast<std::uint32_t>(count);
    return true;
}

std::uint64_t candidate_identity(
    std::uint64_t source_identity,
    const bell_semantic_plan_view &plan,
    const cellpack::local_cell_order_view &order,
    std::uint32_t size) noexcept {
    std::uint64_t identity = fnv_offset;
    const std::uint64_t parts[] = {physical_bell_schema_version, source_identity,
        plan.feature_block_geometry_identity, order.ordering_identity,
        order.row_domain_identity, size};
    for (const std::uint64_t part : parts) hash(&identity, part);
    for (std::uint32_t block = 0u; block <= plan.feature_block_count; ++block)
        hash(&identity, plan.feature_block_offsets[block]);
    for (std::uint32_t feature = 0u; feature < plan.feature_count; ++feature)
        hash(&identity, plan.feature_permutation[feature]);
    for (std::uint32_t row = 0u; row < order.row_count; ++row)
        hash(&identity, order.row_permutation[row]);
    return identity == 0u ? 1u : identity;
}

bell_candidate_requirements evaluate(
    const bell_csr_source_view &source,
    const bell_semantic_plan_view &plan,
    const cellpack::local_cell_order_view &order,
    const bell_lowering_policy &policy,
    const bell_lowering_workspace &workspace,
    std::uint64_t source_identity,
    std::uint32_t size) noexcept {
    bell_candidate_requirements result;
    result.block_size = size; result.row_count = source.row_count;
    result.feature_count = source.feature_count;
    result.feature_block_offset_count = static_cast<std::size_t>(plan.feature_block_count) + 1u;
    std::uint32_t physical_blocks = 0u;
    if (!fill_physical_offsets(plan, size, workspace.feature_block_block_offsets, &physical_blocks))
        return result;
    const std::uint64_t block_rows = (static_cast<std::uint64_t>(source.row_count) + size - 1u) / size;
    const std::uint64_t padded_rows = block_rows * size;
    const std::uint64_t padded_features = static_cast<std::uint64_t>(physical_blocks) * size;
    if (padded_rows > std::numeric_limits<std::uint32_t>::max()
        || padded_features > std::numeric_limits<std::uint32_t>::max()) return result;
    result.block_row_count = static_cast<std::uint32_t>(block_rows);
    result.padded_row_count = static_cast<std::uint32_t>(padded_rows);
    result.padded_feature_count = static_cast<std::uint32_t>(padded_features);

    std::fill(workspace.markers, workspace.markers + physical_blocks, 0u);
    std::uint32_t epoch = 1u, maximum = 0u;
    std::uint64_t occupied = 0u;
    for (std::uint32_t block_row = 0u; block_row < result.block_row_count; ++block_row, ++epoch) {
        std::uint32_t active = 0u;
        const std::uint32_t begin = block_row * size;
        const std::uint32_t end = std::min<std::uint32_t>(source.row_count, begin + size);
        for (std::uint32_t execution_row = begin; execution_row < end; ++execution_row) {
            const std::uint32_t row = order.row_permutation[execution_row];
            for (std::uint32_t entry = source.row_offsets[row]; entry < source.row_offsets[row + 1u]; ++entry) {
                const std::uint32_t execution = plan.inverse_feature_permutation[source.feature_ids[entry]];
                const std::uint32_t physical = physical_block(plan,
                    workspace.feature_block_block_offsets, execution, size);
                if (workspace.markers[physical] != epoch) {
                    workspace.markers[physical] = epoch; ++active;
                }
            }
        }
        occupied += active; maximum = std::max(maximum, active);
    }
    result.ell_blocks_per_row = maximum;
    std::uint64_t stored = 0u, slots = 0u, bytes = 0u, persistent = 0u;
    if (!multiply(block_rows, maximum, &stored)
        || !multiply(stored, size, &slots) || !multiply(slots, size, &slots)
        || !multiply(slots, source.value_size_bytes, &bytes)
        || !multiply(stored, sizeof(std::int32_t), &persistent)
        || persistent > std::numeric_limits<std::uint64_t>::max() - bytes
        || maximum > std::numeric_limits<std::uint32_t>::max() / size
        || stored > std::numeric_limits<std::size_t>::max()
        || bytes > std::numeric_limits<std::size_t>::max()) return result;
    persistent += bytes;
    const std::uint64_t offset_bytes = result.feature_block_offset_count
        * sizeof(std::uint32_t);
    if (persistent > std::numeric_limits<std::uint64_t>::max() - offset_bytes)
        return result;
    persistent += offset_bytes;
    result.ell_columns = maximum * size;
    result.column_index_count = static_cast<std::size_t>(stored);
    result.value_bytes = static_cast<std::size_t>(bytes);
    const std::uint64_t source_bytes = (static_cast<std::uint64_t>(source.row_count) + 1u) * 4u
        + static_cast<std::uint64_t>(source.nnz_count) * (4u + source.value_size_bytes);
    result.metrics = {occupied, stored, slots, persistent, source_bytes,
        slots == 0u ? 0.0 : static_cast<double>(source.nnz_count) / static_cast<double>(slots),
        stored == 0u ? 0.0 : static_cast<double>(occupied) / static_cast<double>(stored),
        source.nnz_count == 0u ? 0.0 : static_cast<double>(slots) / source.nnz_count,
        source_bytes == 0u ? 0.0
            : static_cast<double>(persistent) / static_cast<double>(source_bytes)};
    if (source.nnz_count == 0u) result.state = bell_candidate_state::empty_source;
    else if (result.metrics.value_slot_expansion > policy.maximum_value_slot_expansion)
        result.state = bell_candidate_state::value_expansion_exceeded;
    else if (result.metrics.storage_expansion > policy.maximum_storage_expansion)
        result.state = bell_candidate_state::storage_expansion_exceeded;
    else if (persistent > policy.maximum_persistent_bytes)
        result.state = bell_candidate_state::persistent_bytes_exceeded;
    else result.state = bell_candidate_state::legal;
    result.candidate_identity = candidate_identity(source_identity, plan, order, size);
    return result;
}

} // namespace physical_bell_detail

bell_lowering_status query_bell_candidates_host(
    const bell_csr_source_view &source,
    const bell_semantic_plan_view &plan,
    const cellpack::local_cell_order_view &order,
    const bell_lowering_policy &policy,
    const bell_lowering_workspace &workspace,
    bell_candidate_set *out) noexcept {
    if (out == nullptr || !std::isfinite(policy.maximum_value_slot_expansion)
        || !std::isfinite(policy.maximum_storage_expansion)
        || policy.maximum_value_slot_expansion <= 0.0 || policy.maximum_storage_expansion <= 0.0)
        return {bell_lowering_status_code::invalid_argument, "candidate policy is invalid"};
    std::uint64_t source_identity = 0u;
    const bell_lowering_status status = physical_bell_detail::validate(
        source, plan, order, workspace, &source_identity);
    if (!status) return status;
    bell_candidate_set result;
    constexpr std::uint32_t sizes[physical_bell_candidate_count] = {8u, 16u, 32u};
    for (std::uint32_t index = 0u; index < physical_bell_candidate_count; ++index)
        result.candidates[index] = physical_bell_detail::evaluate(source, plan, order,
            policy, workspace, source_identity, sizes[index]);
    *out = result;
    return {};
}

} // namespace cellerator::compute::math
