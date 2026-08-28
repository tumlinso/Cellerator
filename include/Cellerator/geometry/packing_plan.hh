#pragma once

#include "Cellerator/geometry/evaluator.hh"

#include <memory>

namespace cellpack {

inline constexpr u32 packing_plan_semantic_schema_version = 1u;
inline constexpr u32 feature_block_geometry_identity_version = 1u;

enum class packing_row_domain_kind : u32 {
    unknown = 0u,
    full_dataset_identity = 1u,
    sampled_rows_identity = 2u
};

enum class packing_exact_objective_kind : u32 {
    total_bytes = 1u,
    row_active_block_references = 2u,
    weighted_score = 3u
};

struct packing_plan_identity {
    u64 feature_axis_fingerprint = 0u;
    u32 feature_axis_fingerprint_version = 0u;
    packing_row_domain_kind row_domain_kind = packing_row_domain_kind::unknown;
    u64 row_domain_identity = 0u;
    u64 evaluation_source_identity = 0u;
    u64 sampling_provenance_identity = 0u;
};

struct frozen_evaluation_summary {
    packing_occupancy_totals occupancy{};
    packing_cost_estimate cost{};
    double objective = 0.0;
};

struct frozen_packing_plan_build_view {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    const u32 *feature_permutation = nullptr;
    const u32 *inverse_feature_permutation = nullptr;
    u32 feature_block_count = 0u;
    const u32 *feature_block_offsets = nullptr;
    const u32 *feature_to_block = nullptr;
    const u32 *feature_to_local = nullptr;
    u32 row_group_count = 0u;
    const u32 *row_group_offsets = nullptr;
    u32 maximum_feature_block_width = 0u;
    u32 row_group_width = 0u;
    packing_plan_identity identity{};
    packing_exact_objective_kind objective_kind = packing_exact_objective_kind::row_active_block_references;
    u64 cost_policy_identity = 0u;
    frozen_evaluation_summary baseline{};
    frozen_evaluation_summary final{};
};

struct packing_plan_compatibility {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u64 feature_axis_fingerprint = 0u;
    u32 feature_axis_fingerprint_version = 0u;
    packing_row_domain_kind row_domain_kind = packing_row_domain_kind::unknown;
    u64 row_domain_identity = 0u;
};

class frozen_packing_plan {
public:
    frozen_packing_plan() = default;
    frozen_packing_plan(const frozen_packing_plan &) = delete;
    frozen_packing_plan &operator=(const frozen_packing_plan &) = delete;
    frozen_packing_plan(frozen_packing_plan &&) noexcept = default;
    frozen_packing_plan &operator=(frozen_packing_plan &&) noexcept = default;

    u32 semantic_schema_version() const noexcept { return packing_plan_semantic_schema_version; }
    u32 row_count() const noexcept { return row_count_; }
    u32 feature_count() const noexcept { return feature_count_; }
    u32 feature_block_count() const noexcept { return feature_block_count_; }
    u32 row_group_count() const noexcept { return row_group_count_; }
    u32 maximum_feature_block_width() const noexcept { return maximum_feature_block_width_; }
    u32 row_group_width() const noexcept { return row_group_width_; }
    u64 feature_block_geometry_identity() const noexcept {
        return feature_block_geometry_identity_;
    }
    const u32 *feature_permutation() const noexcept { return feature_permutation_.get(); }
    const u32 *inverse_feature_permutation() const noexcept { return inverse_feature_permutation_.get(); }
    const u32 *feature_block_offsets() const noexcept { return feature_block_offsets_.get(); }
    const u32 *feature_to_block() const noexcept { return feature_to_block_.get(); }
    const u32 *feature_to_local() const noexcept { return feature_to_local_.get(); }
    const u32 *row_group_offsets() const noexcept { return row_group_offsets_.get(); }
    const packing_plan_identity &identity() const noexcept { return identity_; }
    packing_exact_objective_kind objective_kind() const noexcept { return objective_kind_; }
    u64 cost_policy_identity() const noexcept { return cost_policy_identity_; }
    const frozen_evaluation_summary &baseline_evaluation() const noexcept { return baseline_; }
    const frozen_evaluation_summary &final_evaluation() const noexcept { return final_; }

    packing_plan_view view() const noexcept;
    validation_result validate() const;
    validation_result validate_compatibility(const packing_plan_compatibility &expected) const;

private:
    u32 row_count_ = 0u;
    u32 feature_count_ = 0u;
    u32 feature_block_count_ = 0u;
    u32 row_group_count_ = 0u;
    u32 maximum_feature_block_width_ = 0u;
    u32 row_group_width_ = 0u;
    u64 feature_block_geometry_identity_ = 0u;
    std::unique_ptr<u32[]> feature_permutation_;
    std::unique_ptr<u32[]> inverse_feature_permutation_;
    std::unique_ptr<u32[]> feature_block_offsets_;
    std::unique_ptr<u32[]> feature_to_block_;
    std::unique_ptr<u32[]> feature_to_local_;
    std::unique_ptr<u32[]> row_group_offsets_;
    packing_plan_identity identity_{};
    packing_exact_objective_kind objective_kind_ = packing_exact_objective_kind::row_active_block_references;
    u64 cost_policy_identity_ = 0u;
    frozen_evaluation_summary baseline_{};
    frozen_evaluation_summary final_{};

    friend validation_result freeze_packing_plan(
        const frozen_packing_plan_build_view &, frozen_packing_plan *);
};

validation_result freeze_packing_plan(
    const frozen_packing_plan_build_view &source,
    frozen_packing_plan *out);

} // namespace cellpack
