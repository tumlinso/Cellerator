#include "Cellerator/geometry/gating.hh"

#include <cstring>
#include <utility>

namespace cellpack {
namespace {

const packed_region_desc *find_region(const static_plan &plan, u32 region_id) {
    for (const packed_region_desc &region : plan.regions) {
        if (region.region_id == region_id) return &region;
    }
    return nullptr;
}

bool active_for_oracle(
    const packed_region_desc &region,
    oracle_gating_scenario scenario,
    u32 microbatch) {
    const region_role role = static_cast<region_role>(region.role);
    if (role == region_role::discarded) return false;
    switch (scenario) {
    case oracle_gating_scenario::all_regions:
        return true;
    case oracle_gating_scenario::alternating_modules:
        if (role == region_role::residual) return false;
        return (region.module_id & 1u) == (microbatch & 1u);
    case oracle_gating_scenario::conditional_only:
        return role == region_role::conditional;
    case oracle_gating_scenario::dense_tile_only:
        return static_cast<layout_kind>(region.layout) == layout_kind::dense_tile;
    case oracle_gating_scenario::high_residual_skip:
        return role != region_role::residual;
    }
    return false;
}

validation_result validate_plan_regions(const static_plan &plan) {
    validation_result desc_result = validate_plan_desc(plan.desc);
    if (!desc_result) return desc_result;
    return validate_region_sequence(
        plan.regions.data(),
        static_cast<u32>(plan.regions.size()),
        plan.desc.row_count,
        plan.desc.feature_count);
}

} // namespace

const char *oracle_gating_scenario_name(oracle_gating_scenario scenario) {
    switch (scenario) {
    case oracle_gating_scenario::all_regions: return "all_regions";
    case oracle_gating_scenario::alternating_modules: return "alternating_modules";
    case oracle_gating_scenario::conditional_only: return "conditional_only";
    case oracle_gating_scenario::dense_tile_only: return "dense_tile_only";
    case oracle_gating_scenario::high_residual_skip: return "high_residual_skip";
    }
    return "unknown";
}

bool parse_oracle_gating_scenario(const char *name, oracle_gating_scenario *out) {
    if (name == nullptr || out == nullptr) return false;
    const oracle_gating_scenario scenarios[] = {
        oracle_gating_scenario::all_regions,
        oracle_gating_scenario::alternating_modules,
        oracle_gating_scenario::conditional_only,
        oracle_gating_scenario::dense_tile_only,
        oracle_gating_scenario::high_residual_skip
    };
    for (oracle_gating_scenario scenario : scenarios) {
        if (std::strcmp(name, oracle_gating_scenario_name(scenario)) == 0) {
            *out = scenario;
            return true;
        }
    }
    return false;
}

route_mask_view view_route_mask(const route_mask &mask) {
    route_mask_view view;
    view.region_ids = mask.region_ids.data();
    view.region_count = static_cast<u32>(mask.region_ids.size());
    return view;
}

route_tape_view view_route_tape(const route_tape &tape) {
    route_tape_view view;
    view.region_ids = tape.region_ids.data();
    view.region_count = static_cast<u32>(tape.region_ids.size());
    return view;
}

validation_result validate_route_mask(const static_plan &plan, route_mask_view mask) {
    if (mask.region_count != 0u && mask.region_ids == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "route mask region ids are null");
    }
    validation_result plan_result = validate_plan_regions(plan);
    if (!plan_result) return plan_result;

    for (u32 i = 0; i < mask.region_count; ++i) {
        const u32 region_id = mask.region_ids[i];
        const packed_region_desc *region = find_region(plan, region_id);
        if (region == nullptr) {
            return validation_error(validation_code::missing_region, region_id, "route mask references an unknown region");
        }
        if (static_cast<region_role>(region->role) == region_role::discarded) {
            return validation_error(validation_code::invalid_region_role, region_id, "route mask cannot select discarded regions");
        }
        for (u32 prior = 0u; prior < i; ++prior) {
            if (mask.region_ids[prior] == region_id) {
                return validation_error(validation_code::duplicate_id, region_id, "route mask contains a duplicate region id");
            }
        }
    }
    return validation_ok();
}

validation_result build_oracle_route_mask(
    const static_plan &plan,
    oracle_gating_scenario scenario,
    u32 microbatch,
    route_mask *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "route mask output is null");
    }
    route_mask mask;
    mask.region_ids.resize(count_oracle_route_regions(plan, scenario, microbatch));
    route_mask_view view;
    validation_result build_result = build_oracle_route_mask_into(
        plan, scenario, microbatch,
        {{mask.region_ids.data(), mask.region_ids.size(), {}}}, &view);
    if (!build_result) return build_result;
    *out = std::move(mask);
    return validation_ok();
}

u32 count_oracle_route_regions(
    const static_plan &plan,
    oracle_gating_scenario scenario,
    u32 microbatch) {
    u32 count = 0u;
    for (const packed_region_desc &region : plan.regions) {
        count += active_for_oracle(region, scenario, microbatch) ? 1u : 0u;
    }
    return count;
}

validation_result build_oracle_route_mask_into(
    const static_plan &plan,
    oracle_gating_scenario scenario,
    u32 microbatch,
    route_id_storage storage,
    route_mask_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "route mask view output is null");
    }
    validation_result plan_result = validate_plan_regions(plan);
    if (!plan_result) return plan_result;
    const u32 required = count_oracle_route_regions(plan, scenario, microbatch);
    if (required != 0u && storage.region_ids.data == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "route mask storage is null");
    }
    if (storage.region_ids.count < required) {
        return validation_error(validation_code::invalid_offsets, required, "route mask storage capacity is insufficient");
    }
    u32 count = 0u;
    for (const packed_region_desc &region : plan.regions) {
        if (active_for_oracle(region, scenario, microbatch)) {
            storage.region_ids.data[count++] = region.region_id;
        }
    }
    out->region_ids = storage.region_ids.data;
    out->region_count = count;
    return validate_route_mask(plan, *out);
}

validation_result validate_route_mask_matches_oracle(
    const static_plan &plan,
    oracle_gating_scenario scenario,
    u32 microbatch,
    route_mask_view mask) {
    validation_result mask_result = validate_route_mask(plan, mask);
    if (!mask_result) return mask_result;

    route_mask expected;
    validation_result expected_result = build_oracle_route_mask(plan, scenario, microbatch, &expected);
    if (!expected_result) return expected_result;
    if (mask.region_count != static_cast<u32>(expected.region_ids.size())) {
        return validation_error(validation_code::invalid_offsets, mask.region_count, "route mask does not match oracle active-set size");
    }
    for (u32 i = 0; i < mask.region_count; ++i) {
        if (mask.region_ids[i] != expected.region_ids[i]) {
            return validation_error(validation_code::invalid_offsets, i, "route mask does not match oracle active-set order");
        }
    }
    return validation_ok();
}

validation_result record_route_tape(route_mask_view mask, route_tape *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "route tape output is null");
    }
    if (mask.region_count != 0u && mask.region_ids == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "route mask region ids are null");
    }
    route_tape tape;
    tape.region_ids.resize(mask.region_count);
    route_tape_view view;
    validation_result result = record_route_tape_into(
        mask, {{tape.region_ids.data(), tape.region_ids.size(), {}}}, &view);
    if (!result) return result;
    *out = std::move(tape);
    return validation_ok();
}

validation_result record_route_tape_into(
    route_mask_view mask,
    route_id_storage storage,
    route_tape_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "route tape view output is null");
    }
    if (mask.region_count != 0u && mask.region_ids == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "route mask region ids are null");
    }
    if (mask.region_count != 0u && storage.region_ids.data == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "route tape storage is null");
    }
    if (storage.region_ids.count < mask.region_count) {
        return validation_error(validation_code::invalid_offsets, mask.region_count, "route tape storage capacity is insufficient");
    }
    if (mask.region_count != 0u) {
        std::memmove(storage.region_ids.data, mask.region_ids, sizeof(u32) * mask.region_count);
    }
    out->region_ids = storage.region_ids.data;
    out->region_count = mask.region_count;
    return validation_ok();
}

validation_result validate_route_tape_for_replay(
    const static_plan &plan,
    route_mask_view expected_mask,
    route_tape_view tape) {
    validation_result mask_result = validate_route_mask(plan, expected_mask);
    if (!mask_result) return mask_result;
    if (tape.region_count != 0u && tape.region_ids == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "route tape region ids are null");
    }
    if (tape.region_count != expected_mask.region_count) {
        return validation_error(validation_code::invalid_offsets, tape.region_count, "route tape length does not match forward route mask");
    }
    for (u32 i = 0; i < tape.region_count; ++i) {
        if (tape.region_ids[i] != expected_mask.region_ids[i]) {
            return validation_error(validation_code::invalid_offsets, i, "route tape does not replay the forward route mask");
        }
    }
    return validation_ok();
}

} // namespace cellpack
