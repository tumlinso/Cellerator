#include "Cellerator/geometry/optimizer/portfolio_v1.hh"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace cellerator::geometry::optimizer::oracle {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

constexpr std::uint64_t hash_literal(const char* text) noexcept {
    std::uint64_t hash = fnv_offset;
    while (*text != 0) {
        hash ^= static_cast<std::uint8_t>(*text++);
        hash *= fnv_prime;
    }
    return hash;
}

constexpr optimizer_strategy_descriptor_v1 strategies[] = {
    {optimizer_strategy_kind::exact_oracle, 1,
     hash_literal("cellerator.optimizer.exact-oracle.v1"),
     capability_exact_certificate | capability_external_snapshot |
     capability_residual_exactness | capability_unique_contribution_ownership |
     capability_pareto_emission | capability_deterministic_replay,
     0, 0, "exact-oracle"},
    {optimizer_strategy_kind::joint_greedy, 1,
     hash_literal("cellerator.optimizer.joint-greedy.v1"),
     capability_joint_source_destination | capability_incremental_refinement |
     capability_multi_operation_objective | capability_residual_exactness |
     capability_unique_contribution_ownership | capability_pareto_emission |
     capability_deterministic_replay,
     1, 0, "joint-greedy"},
    {optimizer_strategy_kind::sparse_multilevel, 1,
     hash_literal("cellerator.optimizer.sparse-multilevel.v1"),
     capability_joint_source_destination | capability_incremental_refinement |
     capability_multi_operation_objective | capability_residual_exactness |
     capability_sparse_hierarchy | capability_provenance |
     capability_work_window_reuse | capability_unique_contribution_ownership |
     capability_pareto_emission | capability_deterministic_replay,
     2, 0, "sparse-multilevel"},
    {optimizer_strategy_kind::bounded_overlap, 1,
     hash_literal("cellerator.optimizer.bounded-overlap.v1"),
     capability_multi_operation_objective | capability_residual_exactness |
     capability_work_window_reuse | capability_bounded_overlap |
     capability_unique_contribution_ownership | capability_pareto_emission |
     capability_deterministic_replay,
     3, 0, "bounded-overlap"},
};

constexpr std::uint64_t common_evidence =
        evidence_strict_compile | evidence_deterministic_replay |
        evidence_exact_census | evidence_objective_cross_check |
        evidence_sanitizers | evidence_pushed_commit;

constexpr optimizer_strategy_evidence_v1 evidence[] = {
    {optimizer_strategy_kind::exact_oracle, 123, 3070,
     evidence_strict_compile | evidence_deterministic_replay |
     evidence_exact_census | evidence_objective_cross_check |
     evidence_pushed_commit,
     "ebc14f6e4d44efba68f98f505f4f0faea0511223", ""},
    {optimizer_strategy_kind::joint_greedy, 94, 3123,
     common_evidence | evidence_immutable_patch,
     "4504041843f1cf9c2e61cafe8bc80002dece55f9",
     "9789c8d8-12a9-4109-830c-90a03259f40b"},
    {optimizer_strategy_kind::sparse_multilevel, 104, 3134,
     common_evidence | evidence_global_u64_identity | evidence_immutable_patch,
     "2f81e0a6b417667f125ea3c9f247c78c3a53330f",
     "69fa2778-80f1-4b71-abfa-db8161c3daf4"},
    {optimizer_strategy_kind::bounded_overlap, 114, 3130,
     common_evidence | evidence_global_u64_identity | evidence_immutable_patch,
     "3819d9b31f84fa9e141bc47857fd4cf52d575793",
     "91f554dd-47b9-4ff3-a4e2-cfa9ad894738"},
};

static_assert(sizeof(strategies) / sizeof(strategies[0]) == 4);
static_assert(sizeof(evidence) / sizeof(evidence[0]) == 4);

void hash_bytes(
        const void* data,
        std::size_t bytes,
        std::uint64_t* hash) noexcept {
    const auto* values = static_cast<const std::uint8_t*>(data);
    for (std::size_t index = 0; index < bytes; ++index) {
        *hash ^= values[index];
        *hash *= fnv_prime;
    }
}

void hash_u64(std::uint64_t value, std::uint64_t* hash) noexcept {
    for (std::uint32_t byte = 0; byte < 8; ++byte) {
        const std::uint8_t part = static_cast<std::uint8_t>(value >> (byte * 8U));
        hash_bytes(&part, sizeof(part), hash);
    }
}

std::uint64_t required_capabilities(optimizer_strategy_kind kind) noexcept {
    switch (kind) {
        case optimizer_strategy_kind::exact_oracle:
            return capability_exact_certificate | capability_external_snapshot |
                   capability_unique_contribution_ownership;
        case optimizer_strategy_kind::joint_greedy:
            return capability_joint_source_destination |
                   capability_incremental_refinement |
                   capability_multi_operation_objective |
                   capability_residual_exactness;
        case optimizer_strategy_kind::sparse_multilevel:
            return capability_sparse_hierarchy | capability_provenance |
                   capability_work_window_reuse |
                   capability_unique_contribution_ownership;
        case optimizer_strategy_kind::bounded_overlap:
            return capability_bounded_overlap | capability_residual_exactness |
                   capability_unique_contribution_ownership;
    }
    return 0;
}

bool is_hex_sha(const char* text) noexcept {
    for (std::uint32_t index = 0; index < 40; ++index) {
        const char value = text[index];
        if (!((value >= '0' && value <= '9') ||
              (value >= 'a' && value <= 'f'))) return false;
    }
    return text[40] == 0;
}

bool is_uuid_or_empty(const char* text) noexcept {
    if (text[0] == 0) return true;
    for (std::uint32_t index = 0; index < 36; ++index) {
        const char value = text[index];
        if (index == 8 || index == 13 || index == 18 || index == 23) {
            if (value != '-') return false;
        } else if (!((value >= '0' && value <= '9') ||
                     (value >= 'a' && value <= 'f'))) {
            return false;
        }
    }
    return text[36] == 0;
}

}  // namespace

optimizer_portfolio_contract_v1 built_in_optimizer_portfolio_v1() noexcept {
    return {1, 4, strategies, 4, 0, evidence};
}

std::uint64_t hash_optimizer_portfolio_contract_v1(
        const optimizer_portfolio_contract_v1& contract) noexcept {
    if ((contract.strategy_count != 0 && contract.strategies == nullptr) ||
        (contract.evidence_count != 0 && contract.evidence == nullptr)) {
        return 0;
    }
    std::uint64_t hash = fnv_offset;
    hash_u64(contract.version, &hash);
    hash_u64(contract.strategy_count, &hash);
    for (std::uint32_t index = 0; index < contract.strategy_count; ++index) {
        const auto& descriptor = contract.strategies[index];
        hash_u64(static_cast<std::uint32_t>(descriptor.kind), &hash);
        hash_u64(descriptor.contract_version, &hash);
        hash_u64(descriptor.stable_strategy_id, &hash);
        hash_u64(descriptor.capabilities, &hash);
        hash_u64(descriptor.registry_order, &hash);
        hash_bytes(descriptor.stable_name, sizeof(descriptor.stable_name), &hash);
    }
    hash_u64(contract.evidence_count, &hash);
    for (std::uint32_t index = 0; index < contract.evidence_count; ++index) {
        const auto& record = contract.evidence[index];
        hash_u64(static_cast<std::uint32_t>(record.kind), &hash);
        hash_u64(record.terminal_task_number, &hash);
        hash_u64(record.terminal_revision, &hash);
        hash_u64(record.evidence_flags, &hash);
        hash_bytes(record.commit_sha, sizeof(record.commit_sha), &hash);
        hash_bytes(record.patch_artifact_uuid,
                   sizeof(record.patch_artifact_uuid), &hash);
    }
    return hash;
}

optimizer_portfolio_validation validate_optimizer_portfolio_contract_v1(
        const optimizer_portfolio_contract_v1& contract) noexcept {
    optimizer_portfolio_validation validation{};
    if (contract.version != 1 || contract.strategy_count != 4 ||
        contract.evidence_count != 4 || contract.strategies == nullptr ||
        contract.evidence == nullptr) {
        validation.status = optimizer_portfolio_status::invalid_contract;
        return validation;
    }
    std::uint32_t strategy_mask = 0;
    std::uint32_t evidence_mask = 0;
    validation.deterministic_registry_order = true;
    validation.no_promoted_strategy = true;
    for (std::uint32_t index = 0; index < contract.strategy_count; ++index) {
        const auto& descriptor = contract.strategies[index];
        const auto kind = static_cast<std::uint32_t>(descriptor.kind);
        if (kind >= 4 || (strategy_mask & (1U << kind)) != 0) {
            validation.status = kind >= 4
                    ? optimizer_portfolio_status::missing_strategy
                    : optimizer_portfolio_status::duplicate_strategy;
            return validation;
        }
        strategy_mask |= 1U << kind;
        if (descriptor.contract_version != 1 || descriptor.stable_strategy_id == 0 ||
            descriptor.stable_name[0] == 0 || descriptor.registry_order != index) {
            validation.status = optimizer_portfolio_status::invalid_contract;
            return validation;
        }
        const auto required = required_capabilities(descriptor.kind);
        if ((descriptor.capabilities & required) != required ||
            (descriptor.capabilities & capability_deterministic_replay) == 0 ||
            (descriptor.capabilities & capability_pareto_emission) == 0) {
            validation.status = optimizer_portfolio_status::missing_capability;
            return validation;
        }
        ++validation.validated_strategies;
    }
    for (std::uint32_t index = 0; index < contract.evidence_count; ++index) {
        const auto& record = contract.evidence[index];
        const auto kind = static_cast<std::uint32_t>(record.kind);
        if (kind >= 4 || (evidence_mask & (1U << kind)) != 0 ||
            record.terminal_task_number == 0 || record.terminal_revision == 0 ||
            !is_hex_sha(record.commit_sha) ||
            !is_uuid_or_empty(record.patch_artifact_uuid) ||
            (record.evidence_flags & evidence_pushed_commit) == 0) {
            validation.status = optimizer_portfolio_status::invalid_evidence;
            return validation;
        }
        if (record.kind != optimizer_strategy_kind::exact_oracle &&
            ((record.evidence_flags & evidence_immutable_patch) == 0 ||
             record.patch_artifact_uuid[0] == 0)) {
            validation.status = optimizer_portfolio_status::invalid_evidence;
            return validation;
        }
        evidence_mask |= 1U << kind;
        ++validation.validated_evidence;
    }
    if (strategy_mask != 0xFU || evidence_mask != 0xFU) {
        validation.status = optimizer_portfolio_status::missing_strategy;
        return validation;
    }
    validation.contract_fingerprint = hash_optimizer_portfolio_contract_v1(contract);
    if (validation.contract_fingerprint == 0) {
        validation.status = optimizer_portfolio_status::invalid_contract;
        return validation;
    }
    validation.status = optimizer_portfolio_status::success;
    return validation;
}

}  // namespace cellerator::geometry::optimizer::oracle
