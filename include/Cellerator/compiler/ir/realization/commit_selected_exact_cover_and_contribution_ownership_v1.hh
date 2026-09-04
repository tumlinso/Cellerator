#pragma once

#include <Cellerator/compiler/ir/realization/freeze_realization_ir_module_and_target_scopes_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::ir::realization::v1 {

struct partial_contributor_v1 {
    stable_identity_v1 identity{};
    std::int64_t numerator = 1;
    std::int64_t denominator = 1;
};

struct exact_cover_entry_v1 {
    std::uint64_t logical_item = 0u;
    stable_identity_v1 atom{};
    stable_identity_v1 owner{};
    std::vector<stable_identity_v1> halos;
    std::vector<stable_identity_v1> replicas;
    std::vector<partial_contributor_v1> contributors;
    std::uint64_t canonical_recovery = 0u;
};

struct exact_cover_v1 {
    stable_identity_v1 identity{};
    stable_identity_v1 certification_receipt{};
    std::uint64_t logical_item_count = 0u;
    std::vector<exact_cover_entry_v1> entries;
};

enum class exact_cover_status_v1 : std::uint8_t {
    exact = 0u,
    invalid_identity,
    invalid_receipt,
    omitted_item,
    duplicate_item,
    invalid_owner,
    duplicate_replica,
    invalid_contributor,
    invalid_recovery,
    rewrite_changed_cover,
};

[[nodiscard]] exact_cover_status_v1 validate_exact_cover_v1(
    const exact_cover_v1& cover,
    std::string* error = nullptr) noexcept;

[[nodiscard]] exact_cover_status_v1 validate_exact_cover_rewrite_v1(
    const exact_cover_v1& before,
    const exact_cover_v1& after,
    std::string* error = nullptr) noexcept;

} // namespace cellerator::compiler::ir::realization::v1
