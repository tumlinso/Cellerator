#pragma once

#include <Cellerator/compute/operation/candidate_catalog_v2.hh>
#include <Cellerator/compute/operation/relation_algebra.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::operation {

inline constexpr std::uint32_t relation_algebra_catalog_schema_version_v1 = 1u;
inline constexpr std::uint32_t relation_algebra_catalog_entry_count_v1 = 7u;
inline constexpr std::uint32_t relation_algebra_catalog_fragment_count_v1 = 2u;

// These encodings are reserved for an operation-core schema-v2 problem. They
// are deliberately disjoint from the frozen schema-v1 operation_kind values.
// Catalog discovery may carry them, but schema-v1 preparation must reject them.
enum class relation_algebra_operation_kind_v2 : std::uint16_t {
    contract_on_support = 0x1003u,
    segment_reduce = 0x1004u,
    segment_normalize = 0x1005u,
    edge_map_or_gate = 0x1006u,
    relation_bundle_apply = 0x1007u
};

constexpr core::operation_kind operation_core_kind_v2(
    relation_algebra_operation_kind_v2 kind) noexcept {
    return static_cast<core::operation_kind>(
        static_cast<std::uint16_t>(kind));
}

// Relation semantics stay adjacent to the generic candidate descriptor rather
// than being folded into it. Candidate identity joins the two immutable views.
// This leaves candidate-catalog-v2 exact and keeps frozen operation-core v1
// meanings independent of relation-algebra semantics.
struct relation_algebra_catalog_entry_v1 {
    std::uint32_t schema_version = relation_algebra_catalog_schema_version_v1;
    std::uint32_t record_bytes = sizeof(relation_algebra_catalog_entry_v1);
    relation_algebra_kind_v1 relation_kind =
        relation_algebra_kind_v1::relation_apply;
    operation_core_compatibility_v1 compatibility =
        operation_core_compatibility_v1::direct_schema_v1;
    std::uint8_t reserved0[1]{};
    std::uint32_t required_operation_core_schema =
        core::operation_core_schema_version;
    core::stable_id candidate_identity{};
    std::uint32_t reserved[4]{};
};

struct relation_algebra_catalog_view_v1 {
    const relation_algebra_catalog_entry_v1 *entries = nullptr;
    std::uint32_t entry_count = 0u;
    const core::candidate_catalog_fragment_v2 *fragments = nullptr;
    std::uint32_t fragment_count = 0u;
};

relation_algebra_catalog_view_v1 relation_algebra_candidate_catalog_v1() noexcept;

const relation_algebra_catalog_entry_v1 *find_relation_algebra_catalog_entry_v1(
    relation_algebra_kind_v1 kind) noexcept;

const core::candidate_descriptor_v2 *find_relation_algebra_candidate_v2(
    relation_algebra_kind_v1 kind) noexcept;

core::operation_status validate_relation_algebra_candidate_catalog_v1() noexcept;

static_assert(std::is_trivially_copyable<relation_algebra_catalog_entry_v1>::value,
    "relation catalog entries must remain immutable POD metadata");
static_assert(std::is_standard_layout<relation_algebra_catalog_entry_v1>::value,
    "relation catalog entries must remain field-addressable");
static_assert(std::is_trivially_copyable<relation_algebra_catalog_view_v1>::value,
    "relation catalog views must remain non-owning");

} // namespace cellerator::compute::operation
