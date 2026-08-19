#include <Cellerator/compute/math/physical_bell.hh>

#include <algorithm>
#include <cstring>

namespace cellerator::compute::math {
namespace physical_bell_detail {

bool fill_physical_offsets(const bell_semantic_plan_view &, std::uint32_t,
    std::uint32_t *, std::uint32_t *) noexcept;
std::uint32_t physical_block(const bell_semantic_plan_view &, const std::uint32_t *,
    std::uint32_t, std::uint32_t) noexcept;
bell_lowering_status validate_bell_shapes(const bell_csr_source_view &,
    const bell_semantic_plan_view &, const cellpack::local_cell_order_view &) noexcept;

namespace {

void hash(std::uint64_t *value, std::uint64_t item) noexcept {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        *value ^= (item >> (byte * 8u)) & 0xffu;
        *value *= 1099511628211ull;
    }
}

} // namespace

bell_lowering_status validate(
    const bell_csr_source_view &source,
    const bell_semantic_plan_view &plan,
    const cellpack::local_cell_order_view &order,
    const bell_lowering_workspace &workspace,
    std::uint64_t *source_identity) noexcept {
    if (source_identity == nullptr)
        return {bell_lowering_status_code::invalid_argument, "identity output is null"};
    const bell_lowering_status shape_status = validate_bell_shapes(source, plan, order);
    if (!shape_status) return shape_status;
    const std::size_t markers = std::max<std::size_t>(source.row_count, source.feature_count);
    if (workspace.marker_capacity < markers || workspace.markers == nullptr
        || workspace.feature_block_offset_capacity
            < static_cast<std::size_t>(plan.feature_block_count) + 1u
        || workspace.feature_block_block_offsets == nullptr)
        return {bell_lowering_status_code::insufficient_capacity, "workspace is too small"};
    std::fill(workspace.markers, workspace.markers + source.feature_count, 0u);
    for (std::uint32_t execution = 0u; execution < source.feature_count; ++execution) {
        const std::uint32_t canonical = plan.feature_permutation[execution];
        if (canonical >= source.feature_count || workspace.markers[canonical] != 0u
            || plan.inverse_feature_permutation[canonical] != execution)
            return {bell_lowering_status_code::incompatible_plan, "feature maps are not inverses"};
        workspace.markers[canonical] = 1u;
    }
    std::fill(workspace.markers, workspace.markers + source.row_count, 0u);
    for (std::uint32_t execution = 0u; execution < source.row_count; ++execution) {
        const std::uint32_t canonical = order.row_permutation[execution];
        if (canonical >= source.row_count || workspace.markers[canonical] != 0u
            || order.inverse_row_permutation[canonical] != execution)
            return {bell_lowering_status_code::incompatible_order, "row maps are not inverses"};
        workspace.markers[canonical] = 1u;
    }
    if (source.row_offsets[0] != 0u || source.row_offsets[source.row_count] != source.nnz_count)
        return {bell_lowering_status_code::invalid_source, "CSR offsets do not span nnz"};
    std::uint64_t identity = 1469598103934665603ull;
    hash(&identity, source.row_count); hash(&identity, source.feature_count);
    hash(&identity, source.nnz_count); hash(&identity, source.value_size_bytes);
    for (std::uint32_t row = 0u; row < source.row_count; ++row) {
        const std::uint32_t begin = source.row_offsets[row], end = source.row_offsets[row + 1u];
        if (begin > end || end > source.nnz_count)
            return {bell_lowering_status_code::invalid_source, "CSR offsets are not monotone"};
        hash(&identity, begin);
        for (std::uint32_t entry = begin; entry < end; ++entry) {
            const std::uint32_t feature = source.feature_ids[entry];
            if (feature >= source.feature_count
                || (entry != begin && feature <= source.feature_ids[entry - 1u]))
                return {bell_lowering_status_code::invalid_source, "CSR features are not unique and sorted"};
            hash(&identity, feature);
        }
    }
    hash(&identity, source.nnz_count);
    *source_identity = identity == 0u ? 1u : identity;
    return {};
}

} // namespace physical_bell_detail

bell_lowering_status query_bell_lowering_workspace_requirements(
    const bell_csr_source_view &source,
    const bell_semantic_plan_view &plan,
    const cellpack::local_cell_order_view &order,
    bell_lowering_workspace_requirements *out) noexcept {
    if (out == nullptr || source.row_count != order.row_count
        || source.feature_count != plan.feature_count || plan.feature_count == 0u
        || plan.feature_block_count == 0u)
        return {bell_lowering_status_code::invalid_argument, "workspace shapes are incompatible"};
    out->marker_count = std::max<std::size_t>(source.row_count, source.feature_count);
    out->feature_block_offset_count = static_cast<std::size_t>(plan.feature_block_count) + 1u;
    return {};
}

bell_lowering_status materialize_bell_candidate_host(
    const bell_csr_source_view &source,
    const bell_semantic_plan_view &plan,
    const cellpack::local_cell_order_view &order,
    const bell_lowering_policy &policy,
    const bell_candidate_requirements &candidate,
    const bell_lowering_workspace &workspace,
    const bell_candidate_buffers &buffers,
    physical_bell_view *out) noexcept {
    if (out == nullptr) return {bell_lowering_status_code::invalid_argument, "output is null"};
    if (candidate.block_size != 8u && candidate.block_size != 16u
        && candidate.block_size != 32u)
        return {bell_lowering_status_code::candidate_mismatch, "candidate block size is invalid"};
    bell_candidate_set current;
    const bell_lowering_status status = query_bell_candidates_host(
        source, plan, order, policy, workspace, &current);
    if (!status) return status;
    const std::uint32_t index = candidate.block_size == 8u ? 0u
        : (candidate.block_size == 16u ? 1u : 2u);
    const bell_candidate_requirements &shape = current.candidates[index];
    if (candidate.state != bell_candidate_state::legal
        || shape.state != bell_candidate_state::legal)
        return {bell_lowering_status_code::candidate_rejected, "candidate was rejected"};
    if (candidate.candidate_identity != shape.candidate_identity
        || candidate.row_count != shape.row_count
        || candidate.feature_count != shape.feature_count
        || candidate.padded_row_count != shape.padded_row_count
        || candidate.padded_feature_count != shape.padded_feature_count
        || candidate.block_row_count != shape.block_row_count
        || candidate.ell_blocks_per_row != shape.ell_blocks_per_row
        || candidate.ell_columns != shape.ell_columns
        || candidate.feature_block_offset_count != shape.feature_block_offset_count
        || candidate.column_index_count != shape.column_index_count
        || candidate.value_bytes != shape.value_bytes)
        return {bell_lowering_status_code::candidate_mismatch, "candidate geometry is stale"};
    if (buffers.feature_block_offset_capacity < shape.feature_block_offset_count
        || buffers.padded_feature_block_offsets == nullptr
        || buffers.column_index_capacity < shape.column_index_count
        || (shape.column_index_count != 0u && buffers.column_indices == nullptr)
        || buffers.value_capacity_bytes < shape.value_bytes
        || (shape.value_bytes != 0u && buffers.values == nullptr))
        return {bell_lowering_status_code::insufficient_capacity, "output buffers are too small"};

    std::uint32_t physical_blocks = 0u;
    if (!physical_bell_detail::fill_physical_offsets(plan, shape.block_size,
            workspace.feature_block_block_offsets, &physical_blocks))
        return {bell_lowering_status_code::overflow, "physical feature padding overflowed"};
    for (std::uint32_t block = 0u; block <= plan.feature_block_count; ++block)
        buffers.padded_feature_block_offsets[block]
            = workspace.feature_block_block_offsets[block] * shape.block_size;
    std::fill(buffers.column_indices, buffers.column_indices + shape.column_index_count, -1);
    std::memset(buffers.values, 0, shape.value_bytes);
    const auto *src = static_cast<const unsigned char *>(source.values);
    auto *dst = static_cast<unsigned char *>(buffers.values);

    for (std::uint32_t block_row = 0u; block_row < shape.block_row_count; ++block_row) {
        std::fill(workspace.markers, workspace.markers + physical_blocks, 0u);
        const std::uint32_t begin = block_row * shape.block_size;
        const std::uint32_t end = std::min<std::uint32_t>(source.row_count, begin + shape.block_size);
        for (std::uint32_t execution_row = begin; execution_row < end; ++execution_row) {
            const std::uint32_t row = order.row_permutation[execution_row];
            for (std::uint32_t entry = source.row_offsets[row]; entry < source.row_offsets[row + 1u]; ++entry) {
                const std::uint32_t execution = plan.inverse_feature_permutation[source.feature_ids[entry]];
                workspace.markers[physical_bell_detail::physical_block(plan,
                    workspace.feature_block_block_offsets, execution,
                    shape.block_size)] = 1u;
            }
        }
        std::uint32_t slot = 0u;
        for (std::uint32_t physical = 0u; physical < physical_blocks; ++physical) {
            if (workspace.markers[physical] == 0u) continue;
            workspace.markers[physical] = ++slot;
            buffers.column_indices[static_cast<std::size_t>(block_row)
                * shape.ell_blocks_per_row + slot - 1u] = static_cast<std::int32_t>(physical);
        }
        for (std::uint32_t execution_row = begin; execution_row < end; ++execution_row) {
            const std::uint32_t row = order.row_permutation[execution_row];
            for (std::uint32_t entry = source.row_offsets[row]; entry < source.row_offsets[row + 1u]; ++entry) {
                const std::uint32_t execution = plan.inverse_feature_permutation[source.feature_ids[entry]];
                std::uint32_t semantic = 0u;
                while (plan.feature_block_offsets[semantic + 1u] <= execution) ++semantic;
                const std::uint32_t local = execution - plan.feature_block_offsets[semantic];
                const std::uint32_t physical = physical_bell_detail::physical_block(plan,
                    workspace.feature_block_block_offsets, execution,
                    shape.block_size);
                const std::uint64_t block_index = static_cast<std::uint64_t>(block_row)
                    * shape.ell_blocks_per_row + workspace.markers[physical] - 1u;
                const std::uint64_t scalar = block_index * shape.block_size * shape.block_size
                    + (execution_row - begin) * shape.block_size + local % shape.block_size;
                std::memcpy(dst + scalar * source.value_size_bytes,
                    src + static_cast<std::uint64_t>(entry) * source.value_size_bytes,
                    source.value_size_bytes);
            }
        }
    }
    *out = {physical_bell_schema_version, shape.block_size, source.row_count,
        source.feature_count, shape.padded_row_count, shape.padded_feature_count, shape.ell_columns,
        source.value_size_bytes, plan.feature_block_geometry_identity, order.ordering_identity,
        order.row_domain_identity, shape.candidate_identity,
        buffers.padded_feature_block_offsets, buffers.column_indices, buffers.values,
        shape.metrics};
    return {};
}

} // namespace cellerator::compute::math
