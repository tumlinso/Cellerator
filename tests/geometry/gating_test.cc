#include <Cellerator/geometry/gating.hh>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

cellpack::static_plan build_fixture_plan() {
    constexpr cellpack::u32 residual_module = 99u;
    const cellpack::u32 feature_modules[] = {
        0u, 0u, 1u, 1u, 2u, residual_module
    };
    const cellpack::u32 row_offsets[] = {
        0u, 2u, 4u, 6u, 8u, 10u
    };
    const cellpack::u32 row_modules[] = {
        0u, 1u,
        1u, 0u,
        1u, 2u,
        0u, 2u,
        2u, residual_module
    };

    cellpack::feature_module_assignment_view features;
    features.feature_to_module = feature_modules;
    features.feature_count = 6u;
    features.residual_module_id = residual_module;

    cellpack::row_signature_view rows;
    rows.row_count = 5u;
    rows.row_offsets = row_offsets;
    rows.module_ids = row_modules;
    rows.entry_count = 10u;

    cellpack::planner_config config;
    config.residual_module_id = residual_module;
    config.min_primary_rows = 2u;

    cellpack::static_plan plan;
    cellpack::validation_result result = cellpack::build_static_plan(features, rows, config, &plan);
    require(static_cast<bool>(result), result.message);

    for (cellpack::packed_region_desc &region : plan.regions) {
        if (region.module_id == 2u) {
            region.layout = cellpack::to_u32(cellpack::layout_kind::dense_tile);
        }
    }

    cellpack::packed_region_desc discarded{};
    discarded.region_id = static_cast<cellpack::u32>(plan.regions.size());
    discarded.parent_id = cellpack::invalid_id;
    discarded.layout = cellpack::to_u32(cellpack::layout_kind::blocked_ell);
    discarded.role = cellpack::to_u32(cellpack::region_role::discarded);
    discarded.module_id = 77u;
    discarded.row_begin = 0u;
    discarded.row_count = 1u;
    discarded.feature_begin = 0u;
    discarded.feature_count = 1u;
    discarded.index_offset = cellpack::invalid_id;
    discarded.value_offset = cellpack::invalid_id;
    discarded.aux_offset = cellpack::invalid_id;
    discarded.weight_offset = cellpack::invalid_id;
    discarded.output_offset = cellpack::invalid_id;
    plan.regions.push_back(discarded);
    plan.desc.region_count = static_cast<cellpack::u32>(plan.regions.size());
    return plan;
}

std::vector<float> route_reference(
    const cellpack::static_plan &plan,
    const cellpack::route_mask &mask) {
    std::vector<float> y(plan.desc.row_count, 0.0f);
    for (cellpack::u32 region_id : mask.region_ids) {
        for (const cellpack::packed_region_desc &region : plan.regions) {
            if (region.region_id != region_id) continue;
            for (cellpack::u32 row = region.row_begin; row < region.row_begin + region.row_count; ++row) {
                y[row] += static_cast<float>(region.module_id + 1u);
            }
        }
    }
    return y;
}

} // namespace

int main() {
    cellpack::static_plan plan = build_fixture_plan();

    cellpack::route_mask all_regions;
    cellpack::validation_result result = cellpack::build_oracle_route_mask(
        plan,
        cellpack::oracle_gating_scenario::all_regions,
        0u,
        &all_regions);
    require(static_cast<bool>(result), result.message);
    require(!all_regions.region_ids.empty(), "all-regions oracle produced an empty mask");
    for (cellpack::u32 region_id : all_regions.region_ids) {
        const cellpack::packed_region_desc &region = plan.regions[region_id];
        require(region.role != cellpack::to_u32(cellpack::region_role::discarded), "all-regions selected discarded region");
    }

    cellpack::route_mask alternating_even;
    result = cellpack::build_oracle_route_mask(
        plan,
        cellpack::oracle_gating_scenario::alternating_modules,
        0u,
        &alternating_even);
    require(static_cast<bool>(result), result.message);
    for (cellpack::u32 region_id : alternating_even.region_ids) {
        require((plan.regions[region_id].module_id & 1u) == 0u, "alternating oracle selected wrong module parity");
        require(plan.regions[region_id].role != cellpack::to_u32(cellpack::region_role::residual),
                "alternating oracle selected residual region");
    }

    cellpack::route_mask conditional;
    result = cellpack::build_oracle_route_mask(
        plan,
        cellpack::oracle_gating_scenario::conditional_only,
        0u,
        &conditional);
    require(static_cast<bool>(result), result.message);
    require(!conditional.region_ids.empty(), "conditional oracle should find fixture conditional regions");
    for (cellpack::u32 region_id : conditional.region_ids) {
        require(plan.regions[region_id].role == cellpack::to_u32(cellpack::region_role::conditional),
                "conditional oracle selected a non-conditional region");
    }

    cellpack::route_mask dense;
    result = cellpack::build_oracle_route_mask(
        plan,
        cellpack::oracle_gating_scenario::dense_tile_only,
        0u,
        &dense);
    require(static_cast<bool>(result), result.message);
    require(!dense.region_ids.empty(), "dense-tile oracle should find fixture dense regions");
    for (cellpack::u32 region_id : dense.region_ids) {
        require(plan.regions[region_id].layout == cellpack::to_u32(cellpack::layout_kind::dense_tile),
                "dense-tile oracle selected wrong layout");
    }

    cellpack::route_mask no_residual;
    result = cellpack::build_oracle_route_mask(
        plan,
        cellpack::oracle_gating_scenario::high_residual_skip,
        0u,
        &no_residual);
    require(static_cast<bool>(result), result.message);
    for (cellpack::u32 region_id : no_residual.region_ids) {
        require(plan.regions[region_id].role != cellpack::to_u32(cellpack::region_role::residual),
                "high-residual-skip selected residual region");
    }

    cellpack::route_mask duplicate = all_regions;
    duplicate.region_ids.push_back(duplicate.region_ids.front());
    result = cellpack::validate_route_mask(plan, cellpack::view_route_mask(duplicate));
    require(result.code == cellpack::validation_code::duplicate_id, "duplicate route id was accepted");

    cellpack::route_mask unknown;
    unknown.region_ids.push_back(1000u);
    result = cellpack::validate_route_mask(plan, cellpack::view_route_mask(unknown));
    require(result.code == cellpack::validation_code::missing_region, "unknown route id was accepted");

    cellpack::route_mask discarded;
    discarded.region_ids.push_back(static_cast<cellpack::u32>(plan.regions.size() - 1u));
    result = cellpack::validate_route_mask(plan, cellpack::view_route_mask(discarded));
    require(result.code == cellpack::validation_code::invalid_region_role, "discarded route id was accepted");

    cellpack::route_mask residual_misuse = no_residual;
    for (const cellpack::packed_region_desc &region : plan.regions) {
        if (region.role == cellpack::to_u32(cellpack::region_role::residual)) {
            residual_misuse.region_ids.push_back(region.region_id);
            break;
        }
    }
    result = cellpack::validate_route_mask_matches_oracle(
        plan,
        cellpack::oracle_gating_scenario::high_residual_skip,
        0u,
        cellpack::view_route_mask(residual_misuse));
    require(result.code == cellpack::validation_code::invalid_offsets, "residual misuse matched high-residual-skip oracle");

    cellpack::route_tape tape;
    result = cellpack::record_route_tape(cellpack::view_route_mask(alternating_even), &tape);
    require(static_cast<bool>(result), result.message);
    require(tape.region_ids == alternating_even.region_ids, "route tape did not preserve active order");

    cellpack::route_tape second_tape;
    result = cellpack::record_route_tape(cellpack::view_route_mask(alternating_even), &second_tape);
    require(static_cast<bool>(result), result.message);
    require(tape.region_ids == second_tape.region_ids, "route tape is not deterministic");

    cellpack::route_mask alternating_odd;
    result = cellpack::build_oracle_route_mask(
        plan,
        cellpack::oracle_gating_scenario::alternating_modules,
        1u,
        &alternating_odd);
    require(static_cast<bool>(result), result.message);
    result = cellpack::validate_route_tape_for_replay(
        plan,
        cellpack::view_route_mask(alternating_odd),
        cellpack::view_route_tape(tape));
    require(result.code == cellpack::validation_code::invalid_offsets, "wrong route tape was accepted for replay");

    const std::vector<float> full = route_reference(plan, all_regions);
    cellpack::route_mask manual_no_gating;
    for (const cellpack::packed_region_desc &region : plan.regions) {
        if (region.role != cellpack::to_u32(cellpack::region_role::discarded)) {
            manual_no_gating.region_ids.push_back(region.region_id);
        }
    }
    const std::vector<float> manual = route_reference(plan, manual_no_gating);
    require(full.size() == manual.size(), "all-regions reference size mismatch");
    for (std::size_t i = 0; i < full.size(); ++i) {
        require(std::fabs(full[i] - manual[i]) <= 0.0f, "all-regions mask differs from no-gating reference");
    }

    return 0;
}
