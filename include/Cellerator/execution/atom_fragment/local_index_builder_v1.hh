#pragma once

#include <Cellerator/execution/index_space/hierarchical_index_space_v1.hh>

#include <cstdint>

namespace cellerator::execution::atom_fragment {

struct atom_local_component_source_v1 {
    std::uint64_t component_identity = 0u;
    std::uint64_t global_extent = 0u;
    std::uint64_t partition_identity = 0u;
    const std::uint64_t *global_indices = nullptr;
    const std::uint64_t *global_identity_sidecar = nullptr;
    std::uint64_t local_extent = 0u;
};

struct atom_local_index_buffers_v1 {
    hierarchical_index_component_v1 *components = nullptr;
    std::uint64_t component_capacity = 0u;
    std::uint64_t *global_indices = nullptr;
    std::uint64_t global_index_capacity = 0u;
    std::uint64_t *global_identity_sidecar = nullptr;
    std::uint64_t sidecar_capacity = 0u;
};

enum class atom_local_index_build_code_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    insufficient_capacity,
    invalid_component,
    duplicate_or_unordered_component,
    invalid_global_index,
    duplicate_or_unordered_global_index,
    arithmetic_overflow,
};

struct atom_local_index_build_result_v1 {
    atom_local_index_build_code_v1 code = atom_local_index_build_code_v1::success;
    std::uint64_t component_index = 0u;
    std::uint64_t element_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == atom_local_index_build_code_v1::success;
    }
};

atom_local_index_build_result_v1 build_atom_local_index_space_v1(
    std::uint64_t relation_identity, std::uint64_t aggregate_extent,
    const atom_local_component_source_v1 *sources, std::uint64_t source_count,
    const atom_local_index_buffers_v1 &buffers,
    hierarchical_index_space_view_v1 *result) noexcept;

} // namespace cellerator::execution::atom_fragment
