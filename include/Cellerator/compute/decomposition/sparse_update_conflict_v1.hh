#pragma once

#include <Cellerator/compute/operation/relation_algebra_v2/composition.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t sparse_update_conflict_schema_version_v1 = 1u;

enum class sparse_update_conflict_policy_v1 : std::uint8_t {
    unique_proven = 1u,
    reject_conflicts = 2u,
    deterministic_serial = 3u,
    stable_grouped_reduce = 4u,
    atomic_unordered = 5u
};

enum class sparse_update_order_v1 : std::uint8_t {
    none = 1u,
    stable_input = 2u,
    stable_fragment_then_input = 3u,
    canonical_index_then_input = 4u
};

struct sparse_update_conflict_contract_v1 {
    std::uint32_t schema_version = sparse_update_conflict_schema_version_v1;
    std::uint32_t reserved = 0u;
    operation::v2::stable_id identity{};
    const operation::v2::sparse_axis_update_descriptor *update = nullptr;
    std::uint64_t fragment_count = 0u;
    sparse_update_conflict_policy_v1 conflict_policy =
        sparse_update_conflict_policy_v1::unique_proven;
    sparse_update_order_v1 order = sparse_update_order_v1::none;
    bool conflicts_possible = false;
    bool deterministic_required = true;
    bool preserves_all_updates = true;
    std::uint8_t reserved2[3]{};
};

enum class sparse_update_conflict_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    invalid_identity,
    missing_update,
    invalid_update,
    invalid_fragment_count,
    invalid_policy,
    invalid_order,
    unique_conflict_mismatch,
    invalid_unique_policy,
    conflicts_not_handled,
    ordered_operation_requires_stable_order,
    unordered_policy_not_legal,
    deterministic_order_required,
    update_loss_not_permitted
};

struct sparse_update_conflict_validation_result_v1 {
    sparse_update_conflict_validation_code_v1 code =
        sparse_update_conflict_validation_code_v1::ok;

    constexpr explicit operator bool() const noexcept {
        return code == sparse_update_conflict_validation_code_v1::ok;
    }
};

sparse_update_conflict_validation_result_v1
validate_sparse_update_conflict_contract_v1(
    const sparse_update_conflict_contract_v1 &contract) noexcept;

static_assert(
    std::is_trivially_copyable_v<sparse_update_conflict_contract_v1>);
static_assert(std::is_standard_layout_v<sparse_update_conflict_contract_v1>);
static_assert(std::is_trivially_copyable_v<
    sparse_update_conflict_validation_result_v1>);

}  // namespace cellerator::compute::decomposition
