#pragma once

#include <cstdint>

namespace cellerator::geometry::optimizer::oracle {

// Frozen availability contract for the validated joint-optimizer portfolio.
// Registry order is deterministic evidence order only. Planner policy remains
// the sole authority for candidate selection and performance promotion.
enum class optimizer_strategy_kind : std::uint32_t {
    exact_oracle = 0,
    joint_greedy = 1,
    sparse_multilevel = 2,
    bounded_overlap = 3,
};

enum optimizer_strategy_capability : std::uint64_t {
    capability_exact_certificate = 1ULL << 0U,
    capability_external_snapshot = 1ULL << 1U,
    capability_joint_source_destination = 1ULL << 2U,
    capability_incremental_refinement = 1ULL << 3U,
    capability_multi_operation_objective = 1ULL << 4U,
    capability_residual_exactness = 1ULL << 5U,
    capability_sparse_hierarchy = 1ULL << 6U,
    capability_provenance = 1ULL << 7U,
    capability_work_window_reuse = 1ULL << 8U,
    capability_bounded_overlap = 1ULL << 9U,
    capability_unique_contribution_ownership = 1ULL << 10U,
    capability_pareto_emission = 1ULL << 11U,
    capability_deterministic_replay = 1ULL << 12U,
};

struct optimizer_strategy_descriptor_v1 {
    optimizer_strategy_kind kind = optimizer_strategy_kind::exact_oracle;
    std::uint32_t contract_version = 1;
    std::uint64_t stable_strategy_id = 0;
    std::uint64_t capabilities = 0;
    std::uint32_t registry_order = 0;
    std::uint32_t reserved = 0;
    char stable_name[32]{};
};

struct optimizer_strategy_evidence_v1 {
    optimizer_strategy_kind kind = optimizer_strategy_kind::exact_oracle;
    std::uint32_t terminal_task_number = 0;
    std::uint64_t terminal_revision = 0;
    std::uint64_t evidence_flags = 0;
    char commit_sha[41]{};
    char patch_artifact_uuid[37]{};
};

struct optimizer_portfolio_contract_v1 {
    std::uint32_t version = 1;
    std::uint32_t strategy_count = 0;
    const optimizer_strategy_descriptor_v1* strategies = nullptr;
    std::uint32_t evidence_count = 0;
    std::uint32_t reserved = 0;
    const optimizer_strategy_evidence_v1* evidence = nullptr;
};

enum optimizer_evidence_flag : std::uint64_t {
    evidence_strict_compile = 1ULL << 0U,
    evidence_deterministic_replay = 1ULL << 1U,
    evidence_exact_census = 1ULL << 2U,
    evidence_objective_cross_check = 1ULL << 3U,
    evidence_sanitizers = 1ULL << 4U,
    evidence_global_u64_identity = 1ULL << 5U,
    evidence_pushed_commit = 1ULL << 6U,
    evidence_immutable_patch = 1ULL << 7U,
};

enum class optimizer_portfolio_status : std::uint32_t {
    success = 0,
    invalid_argument,
    invalid_contract,
    missing_strategy,
    duplicate_strategy,
    missing_capability,
    invalid_evidence,
};

struct optimizer_portfolio_validation {
    optimizer_portfolio_status status = optimizer_portfolio_status::invalid_argument;
    std::uint64_t contract_fingerprint = 0;
    std::uint32_t validated_strategies = 0;
    std::uint32_t validated_evidence = 0;
    bool deterministic_registry_order = false;
    bool no_promoted_strategy = false;
};

// Static, allocation-free, cold registry. It records validated availability;
// planner policy remains solely responsible for selection and promotion.
optimizer_portfolio_contract_v1 built_in_optimizer_portfolio_v1() noexcept;

std::uint64_t hash_optimizer_portfolio_contract_v1(
        const optimizer_portfolio_contract_v1& contract) noexcept;

optimizer_portfolio_validation validate_optimizer_portfolio_contract_v1(
        const optimizer_portfolio_contract_v1& contract) noexcept;

}  // namespace cellerator::geometry::optimizer::oracle
