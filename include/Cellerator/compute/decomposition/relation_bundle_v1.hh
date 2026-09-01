#pragma once

#include <Cellerator/compute/decomposition/vocabulary_v1.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t relation_bundle_schema_version_v1 = 1u;

// Relation type is defined by biological endpoint domains, never by shape.
// Each fragment owns a contiguous, nonempty range of the bundle's relations.
struct relation_bundle_type_fragment_v1 {
    std::uint64_t first_relation = 0u;
    std::uint64_t relation_count = 0u;
    execution::domain_id source_domain{};
    execution::domain_id destination_domain{};
};

struct relation_bundle_type_decomposition_v1 {
    std::uint32_t schema_version = relation_bundle_schema_version_v1;
    std::uint32_t reserved = 0u;
    operation::v2::stable_id decomposition_identity{};
    const operation::v2::operation_problem *problem = nullptr;
    const relation_bundle_type_fragment_v1 *fragments = nullptr;
    std::uint64_t fragment_count = 0u;
    decomposition_kind_v1 kind = decomposition_kind_v1::disjoint;
    fragment_role_v1 fragment_role = fragment_role_v1::owned;
    bool produces_partial_results = true;
    bool requires_partial_algebra = true;
    std::uint8_t reserved2[4]{};
};

enum class relation_bundle_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    invalid_identity,
    missing_problem,
    invalid_problem,
    unsupported_operation,
    missing_fragments,
    invalid_fragment_count,
    invalid_vocabulary,
    invalid_partial_result_contract,
    empty_fragment,
    relation_offset_mismatch,
    relation_range_overflow,
    invalid_relation_type,
    relation_type_mismatch,
    incomplete_relation_partition
};

struct relation_bundle_validation_result_v1 {
    relation_bundle_validation_code_v1 code =
        relation_bundle_validation_code_v1::ok;
    std::uint64_t fragment_index = 0u;
    std::uint64_t relation_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == relation_bundle_validation_code_v1::ok;
    }
};

relation_bundle_validation_result_v1
validate_relation_bundle_type_decomposition_v1(
    const relation_bundle_type_decomposition_v1 &decomposition) noexcept;

static_assert(std::is_trivially_copyable_v<relation_bundle_type_fragment_v1>);
static_assert(std::is_standard_layout_v<relation_bundle_type_fragment_v1>);
static_assert(
    std::is_trivially_copyable_v<relation_bundle_type_decomposition_v1>);
static_assert(std::is_standard_layout_v<relation_bundle_type_decomposition_v1>);
static_assert(std::is_trivially_copyable_v<relation_bundle_validation_result_v1>);

}  // namespace cellerator::compute::decomposition
