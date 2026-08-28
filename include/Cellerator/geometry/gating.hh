#pragma once

#include "Cellerator/geometry/planner.hh"

#include <vector>

namespace cellpack {

enum class oracle_gating_scenario : u32 {
    all_regions = 0u,
    alternating_modules = 1u,
    conditional_only = 2u,
    dense_tile_only = 3u,
    high_residual_skip = 4u
};

struct route_mask_view {
    const u32 *region_ids = nullptr;
    u32 region_count = 0u;
};

struct route_mask {
    std::vector<u32> region_ids;
};

struct route_tape_view {
    const u32 *region_ids = nullptr;
    u32 region_count = 0u;
};

struct route_tape {
    std::vector<u32> region_ids;
};

struct oracle_active_set_view {
    oracle_gating_scenario scenario = oracle_gating_scenario::all_regions;
    u32 microbatch = 0u;
    const u32 *region_ids = nullptr;
    u32 region_count = 0u;
};

const char *oracle_gating_scenario_name(oracle_gating_scenario scenario);

bool parse_oracle_gating_scenario(const char *name, oracle_gating_scenario *out);

route_mask_view view_route_mask(const route_mask &mask);

route_tape_view view_route_tape(const route_tape &tape);

validation_result validate_route_mask(const static_plan &plan, route_mask_view mask);

validation_result build_oracle_route_mask(
    const static_plan &plan,
    oracle_gating_scenario scenario,
    u32 microbatch,
    route_mask *out);

validation_result validate_route_mask_matches_oracle(
    const static_plan &plan,
    oracle_gating_scenario scenario,
    u32 microbatch,
    route_mask_view mask);

validation_result record_route_tape(route_mask_view mask, route_tape *out);

validation_result validate_route_tape_for_replay(
    const static_plan &plan,
    route_mask_view expected_mask,
    route_tape_view tape);

} // namespace cellpack
