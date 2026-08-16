#pragma once

#include "CellPack/format.hh"
#include "CellPack/validate.hh"

#include <vector>

namespace cellpack {

struct feature_module_assignment_view {
    const u32 *feature_to_module = nullptr;
    u32 feature_count = 0u;
    u32 residual_module_id = invalid_id;
};

struct row_signature_view {
    u32 row_count = 0u;
    const u32 *row_offsets = nullptr;
    const u32 *module_ids = nullptr;
    u32 entry_count = 0u;
};

struct planner_config {
    u32 residual_module_id = invalid_id;
    u32 min_primary_rows = 1u;
    layout_kind primary_layout = layout_kind::blocked_ell;
    layout_kind residual_layout = layout_kind::residual_csr;
    bool emit_residual_region = true;
};

struct static_plan {
    plan_desc desc{};
    std::vector<u32> row_permutation;
    std::vector<u32> inverse_row_permutation;
    std::vector<u32> feature_permutation;
    std::vector<u32> inverse_feature_permutation;
    std::vector<u32> signature_offsets;
    std::vector<u32> signature_module_ids;
    std::vector<feature_module_desc> modules;
    std::vector<row_group_desc> row_groups;
    std::vector<packed_region_desc> regions;
    // Execution-axis boundaries derived from modules and row groups. These are
    // geometry, not a physical sparse-format commitment.
    std::vector<u32> feature_block_offsets;
    std::vector<u32> row_group_offsets;
};

validation_result build_static_plan(
    const feature_module_assignment_view &features,
    const row_signature_view &rows,
    const planner_config &config,
    static_plan *out);

} // namespace cellpack
