#include <Cellerator/execution/atom_fragment/local_index_builder_v1.hh>

#include <limits>

namespace cellerator::execution::atom_fragment {
namespace {

local_index_width_v1 width_for(std::uint64_t count) noexcept {
    if (count <= std::numeric_limits<std::uint16_t>::max())
        return local_index_width_v1::u16;
    if (count <= std::numeric_limits<std::uint32_t>::max())
        return local_index_width_v1::u32;
    return local_index_width_v1::u64;
}

} // namespace

atom_local_index_build_result_v1 build_atom_local_index_space_v1(
    std::uint64_t relation_identity, std::uint64_t aggregate_extent,
    const atom_local_component_source_v1 *sources, std::uint64_t source_count,
    const atom_local_index_buffers_v1 &buffers,
    hierarchical_index_space_view_v1 *result) noexcept {
    if (result == nullptr || relation_identity == 0u || aggregate_extent == 0u
        || sources == nullptr || source_count == 0u) {
        return {atom_local_index_build_code_v1::invalid_argument, 0u, 0u};
    }
    *result = {};
    std::uint64_t total = 0u;
    bool needs_sidecar = false;
    for (std::uint64_t index = 0u; index < source_count; ++index) {
        const auto &source = sources[index];
        if (source.component_identity == 0u || source.partition_identity == 0u
            || source.global_extent == 0u || source.local_extent == 0u
            || source.global_indices == nullptr) {
            return {atom_local_index_build_code_v1::invalid_component,
                index, 0u};
        }
        if (index != 0u
            && sources[index - 1u].component_identity
                >= source.component_identity) {
            return {atom_local_index_build_code_v1::
                duplicate_or_unordered_component, index, 0u};
        }
        if (source.local_extent
            > std::numeric_limits<std::uint64_t>::max() - total) {
            return {atom_local_index_build_code_v1::arithmetic_overflow,
                index, 0u};
        }
        total += source.local_extent;
        needs_sidecar = needs_sidecar
            || source.global_identity_sidecar != nullptr;
        for (std::uint64_t item = 0u; item < source.local_extent; ++item) {
            if (source.global_indices[item] >= source.global_extent) {
                return {atom_local_index_build_code_v1::invalid_global_index,
                    index, item};
            }
            if (item != 0u
                && source.global_indices[item - 1u]
                    >= source.global_indices[item]) {
                return {atom_local_index_build_code_v1::
                    duplicate_or_unordered_global_index, index, item};
            }
        }
    }
    if (total > aggregate_extent) {
        return {atom_local_index_build_code_v1::invalid_component, 0u, total};
    }
    if (buffers.components == nullptr
        || buffers.component_capacity < source_count
        || buffers.global_indices == nullptr
        || buffers.global_index_capacity < total
        || (needs_sidecar && (buffers.global_identity_sidecar == nullptr
            || buffers.sidecar_capacity < total))) {
        return {atom_local_index_build_code_v1::insufficient_capacity, 0u, total};
    }

    std::uint64_t offset = 0u;
    for (std::uint64_t index = 0u; index < source_count; ++index) {
        const auto &source = sources[index];
        auto &component = buffers.components[index];
        component = {};
        component.component_identity = source.component_identity;
        component.aggregate_begin = offset;
        component.index_space.global_extent = source.global_extent;
        component.index_space.partition_identity = source.partition_identity;
        component.index_space.local_extent = source.local_extent;
        component.index_space.local_to_global = buffers.global_indices + offset;
        component.index_space.local_width = width_for(source.local_extent);
        for (std::uint64_t item = 0u; item < source.local_extent; ++item) {
            buffers.global_indices[offset + item] = source.global_indices[item];
        }
        if (source.global_identity_sidecar != nullptr) {
            component.index_space.global_identity_sidecar =
                buffers.global_identity_sidecar + offset;
            for (std::uint64_t item = 0u; item < source.local_extent; ++item) {
                buffers.global_identity_sidecar[offset + item] =
                    source.global_identity_sidecar[item];
            }
        }
        offset += source.local_extent;
    }
    result->relation_identity = relation_identity;
    result->aggregate_extent = aggregate_extent;
    result->components = buffers.components;
    result->component_count = source_count;
    return {};
}

} // namespace cellerator::execution::atom_fragment
