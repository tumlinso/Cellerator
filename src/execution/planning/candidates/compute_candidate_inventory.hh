#pragma once

#include <stdint.h>

namespace cellerator::execution::planning {

struct compute_candidate_inventory_v1 {
    uint64_t relation_apply_candidates = 0u;
    uint64_t residual_candidates = 0u;
    uint64_t contraction_candidates = 0u;
    uint64_t transpose_candidates = 0u;
    uint64_t gate_and_update_candidates = 0u;
    uint64_t bundle_and_chain_candidates = 0u;
    uint64_t segment_candidates = 0u;
    uint64_t fusion_candidates = 0u;
    uint64_t training_stage_kinds = 0u;
    bool all_candidates_planner_owned = false;
    bool all_experimental_candidates_require_measurement = false;
};

compute_candidate_inventory_v1 query_compute_candidate_inventory_v1() noexcept;

} // namespace cellerator::execution::planning
