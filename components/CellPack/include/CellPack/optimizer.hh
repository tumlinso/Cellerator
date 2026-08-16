#pragma once

#include "CellPack/candidate_relation.hh"
#include "CellPack/packing_plan.hh"

#include <cstddef>

namespace cellerator::compute::gene_support {
struct gene_support_bitset_view;
}
namespace cellerator::compute::sampling {
struct sample_provenance;
}

namespace cellpack {

inline constexpr u32 sampled_feature_support_identity_version = 1u;

struct sampled_feature_support_view {
    u32 sampled_row_count = 0u;
    u32 feature_count = 0u;
    u32 words_per_feature = 0u;
    const u32 *support_words = nullptr;
    const u32 *detected_row_counts = nullptr;
    const u64 *sampled_position_to_global_row = nullptr;
    const ::cellerator::compute::sampling::sample_provenance *provenance = nullptr;
};

validation_result make_sampled_feature_support_view(
    const ::cellerator::compute::gene_support::gene_support_bitset_view &source,
    sampled_feature_support_view *out);

validation_result validate_sampled_feature_support_view(const sampled_feature_support_view &support);

// Deterministic identity of the selection contract and exact sampled-position
// to global-row mapping. It deliberately excludes expression/support contents.
validation_result query_sampled_feature_support_identity(
    const sampled_feature_support_view &support,
    u64 *out);

struct packing_optimizer_workspace_view {
    packing_evaluation_workspace_view evaluator_workspace{};
    packing_occupancy_buffers occupancy_buffers{};
};

struct packing_optimizer_workspace_requirements {
    packing_evaluation_requirements evaluator{};
};

enum class packing_optimizer_phase : u32 {
    none = 0u,
    candidate_normalization = 1u,
    baseline = 2u,
    coarsening = 3u,
    refinement = 4u,
    final_verification = 5u,
    freeze = 6u
};

struct packing_optimizer_config {
    u32 maximum_feature_block_width = 16u;
    u32 row_group_width = 128u;
    u32 candidate_fanout = 8u;
    u32 proposal_shortlist = 256u;
    u32 initial_oracle_batch_size = 8u;
    u32 maximum_coarsening_passes = 32u;
    u32 maximum_refinement_passes = 8u;
    u32 maximum_oracle_evaluations = 256u;
    bool enable_feature_moves = true;
    bool enable_feature_swaps = true;
    packing_exact_objective_kind objective_kind = packing_exact_objective_kind::row_active_block_references;
    double weighted_score_absolute_tolerance = 1.0e-9;
    double weighted_score_relative_tolerance = 1.0e-12;
    packing_cost_model cost_model{};
    u64 cost_policy_identity = 0u;
    packing_plan_identity plan_identity{};
};

struct packing_optimizer_diagnostics {
    candidate_normalization_statistics candidate_normalization{};
    u32 initial_block_count = 0u;
    u32 final_block_count = 0u;
    u32 coarsening_passes = 0u;
    u32 refinement_passes = 0u;
    u64 merge_proposals_considered = 0u;
    u64 merge_proposals_shortlisted = 0u;
    u64 merge_proxy_positive = 0u;
    u64 merge_oracle_accepted = 0u;
    u64 merge_oracle_rejected = 0u;
    u64 move_proposals_considered = 0u;
    u64 move_proposals_shortlisted = 0u;
    u64 move_proxy_positive = 0u;
    u64 move_oracle_accepted = 0u;
    u64 move_oracle_rejected = 0u;
    u64 swap_proposals_considered = 0u;
    u64 swap_proposals_shortlisted = 0u;
    u64 swap_proxy_positive = 0u;
    u64 swap_oracle_accepted = 0u;
    u64 swap_oracle_rejected = 0u;
    u64 oracle_evaluations = 0u;
    u64 oracle_rollbacks = 0u;
    u64 oracle_batch_reductions = 0u;
    u64 blacklisted_mutations = 0u;
    double candidate_processing_ms = 0.0;
    double proxy_ms = 0.0;
    double oracle_ms = 0.0;
    double freeze_ms = 0.0;
    double total_ms = 0.0;
    std::size_t peak_additional_optimizer_bytes = 0u;
    packing_optimizer_phase final_phase = packing_optimizer_phase::none;
    frozen_evaluation_summary baseline{};
    frozen_evaluation_summary final{};
};

struct packing_optimizer_result {
    frozen_packing_plan plan;
    packing_optimizer_diagnostics diagnostics{};
};

validation_result query_packing_optimizer_workspace_requirements(
    const prepared_csr_support &source,
    u32 row_group_width,
    packing_optimizer_workspace_requirements *out);

validation_result optimize_packing_plan(
    const prepared_csr_support &source,
    const sampled_feature_support_view &sampled_support,
    candidate_relation_view candidates,
    const packing_optimizer_config &config,
    const packing_optimizer_workspace_view &workspace,
    packing_optimizer_result *out);

} // namespace cellpack
