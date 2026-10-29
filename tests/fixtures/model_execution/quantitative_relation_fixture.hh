#pragma once

#include <Cellerator/execution/identity_registry.hh>
#include <Cellerator/execution/lifetimes.hh>
#include <Cellerator/execution/operands.hh>

#include <cstddef>
#include <cstdint>

namespace cellerator::ce_live {

enum class quantitative_fixture_status : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    invalid_identity = 2u,
    invalid_support = 3u,
    registry_failure = 4u
};

struct quantitative_fixture_identities {
    execution::domain_id feature_domain;
    execution::domain_id observation_domain;
    execution::order_id feature_order;
    execution::order_id observation_order;
    execution::geometry_id geometry;
    execution::partition_id partition;
    execution::structure_id structure;
    execution::projection_id destination_row_csr_projection;
};

// Physical rows are destinations (cells); column indices are logical sources
// (features). This storage choice never reverses the logical relation.
struct destination_row_csr_view {
    const std::uint64_t *destination_offsets;
    const std::uint32_t *source_indices;
    std::uint32_t destination_count;
    std::uint32_t source_count;
    std::uint64_t logical_edge_count;
};

struct quantitative_fixture_arrays {
    destination_row_csr_view support;
    float *generation_1_values;
    float *generation_2_values;
};

struct native_quantitative_relation {
    destination_row_csr_view projection;
    execution::relation_structure structure;
    execution::sparse_relation_view operand;
    execution::value_plane generations[2];
};

// Convert the first 128 bits of a lowercase SHA-256 identity to the stable
// Cellerator persistent-identity width. The byte order is defined here rather
// than inherited from host integer layout.
quantitative_fixture_status identity_from_sha256(
    const char *hex, std::uint64_t *low, std::uint64_t *high) noexcept;

// Exact 128-bit Cellerator projections of the checksum-pinned manifest's
// SHA-256 identities. Large arrays remain external fixture artifacts.
quantitative_fixture_identities pbmc3k_quantitative_v1_identities() noexcept;

quantitative_fixture_status bind_quantitative_fixture(
    const quantitative_fixture_arrays &arrays,
    const quantitative_fixture_identities &identities,
    execution::identity_registry *registry,
    execution::projection_catalog_handle projection_catalog,
    native_quantitative_relation *relation) noexcept;

float deterministic_dense_operand(
    std::uint32_t source, std::uint32_t lane) noexcept;

void fill_deterministic_dense_operand(float *values,
    std::uint32_t source_count, std::uint32_t dense_width) noexcept;

} // namespace cellerator::ce_live
