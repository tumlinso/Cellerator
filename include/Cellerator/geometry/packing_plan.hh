#pragma once

#include "Cellerator/geometry/evaluator.hh"

#include <Cellerator/memory/allocation.hh>
#include <Cellerator/memory/image.hh>

#include <memory>
#include <type_traits>

namespace cellpack {

inline constexpr u32 packing_plan_semantic_schema_version = 1u;
inline constexpr u32 feature_block_geometry_identity_version = 1u;
inline constexpr u32 packing_plan_image_magic = 0x31495043u; // "CPI1"
inline constexpr u32 packing_plan_image_alignment = 64u;

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

struct alignas(64) packing_plan_image_header {
    cellerator::memory::image_header common{};
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 feature_block_count = 0u;
    u32 row_group_count = 0u;
    u32 maximum_feature_block_width = 0u;
    u32 row_group_width = 0u;
    packing_exact_objective_kind objective_kind = packing_exact_objective_kind::row_active_block_references;
    u32 reserved = 0u;
    u64 cost_policy_identity = 0u;
    packing_plan_identity identity{};
    frozen_evaluation_summary baseline{};
    frozen_evaluation_summary final{};
    cellerator::memory::rel32 feature_permutation{};
    cellerator::memory::rel32 inverse_feature_permutation{};
    cellerator::memory::rel32 feature_block_offsets{};
    cellerator::memory::rel32 feature_to_block{};
    cellerator::memory::rel32 feature_to_local{};
    cellerator::memory::rel32 row_group_offsets{};
};

struct packing_plan_image_view {
    packing_plan_image_header header{};
    const void *image_base = nullptr;
    std::size_t image_bytes = 0u;
    const u32 *feature_permutation = nullptr;
    const u32 *inverse_feature_permutation = nullptr;
    const u32 *feature_block_offsets = nullptr;
    const u32 *feature_to_block = nullptr;
    const u32 *feature_to_local = nullptr;
    const u32 *row_group_offsets = nullptr;
};

validation_result validate_packing_plan_image_host(
    const void *image,
    std::size_t image_bytes,
    packing_plan_image_view *out) noexcept;

// Rebinds a previously validated image view to an equal-sized copy. The new
// base may be device memory and is never dereferenced by this operation.
validation_result rebind_packing_plan_image(
    const packing_plan_image_view &validated_host_view,
    const void *new_image_base,
    std::size_t new_image_bytes,
    packing_plan_image_view *out) noexcept;

struct packing_plan_image_deleter {
    void operator()(unsigned char *pointer) const noexcept;
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
    const u32 *feature_permutation() const noexcept { return image_view_.feature_permutation; }
    const u32 *inverse_feature_permutation() const noexcept { return image_view_.inverse_feature_permutation; }
    const u32 *feature_block_offsets() const noexcept { return image_view_.feature_block_offsets; }
    const u32 *feature_to_block() const noexcept { return image_view_.feature_to_block; }
    const u32 *feature_to_local() const noexcept { return image_view_.feature_to_local; }
    const u32 *row_group_offsets() const noexcept { return image_view_.row_group_offsets; }
    const packing_plan_identity &identity() const noexcept { return identity_; }
    packing_exact_objective_kind objective_kind() const noexcept { return objective_kind_; }
    u64 cost_policy_identity() const noexcept { return cost_policy_identity_; }
    const frozen_evaluation_summary &baseline_evaluation() const noexcept { return baseline_; }
    const frozen_evaluation_summary &final_evaluation() const noexcept { return final_; }
    const packing_plan_image_view &image_view() const noexcept { return image_view_; }
    const cellerator::memory::allocation &image_allocation() const noexcept { return image_allocation_; }

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
    std::unique_ptr<unsigned char, packing_plan_image_deleter> image_storage_;
    cellerator::memory::allocation image_allocation_{};
    packing_plan_image_view image_view_{};
    packing_plan_identity identity_{};
    packing_exact_objective_kind objective_kind_ = packing_exact_objective_kind::row_active_block_references;
    u64 cost_policy_identity_ = 0u;
    frozen_evaluation_summary baseline_{};
    frozen_evaluation_summary final_{};

    friend validation_result freeze_packing_plan(
        const frozen_packing_plan_build_view &, frozen_packing_plan *);
};

static_assert(std::is_trivially_copyable<packing_plan_image_header>::value,
    "packing plan image header must remain pointer-free");
static_assert(std::is_trivially_copyable<packing_plan_image_view>::value,
    "packing plan rebound view must remain device-copyable");

validation_result freeze_packing_plan(
    const frozen_packing_plan_build_view &source,
    frozen_packing_plan *out);

} // namespace cellpack
