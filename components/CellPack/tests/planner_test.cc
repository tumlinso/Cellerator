#include <CellPack/planner.hh>

#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

const cellpack::packed_region_desc *find_region(
    const cellpack::static_plan &plan,
    cellpack::u32 module_id,
    cellpack::region_role role) {
    for (const cellpack::packed_region_desc &region : plan.regions) {
        if (region.module_id == module_id && region.role == cellpack::to_u32(role)) return &region;
    }
    return nullptr;
}

} // namespace

int main() {
    constexpr cellpack::u32 residual_module = 99u;
    const cellpack::u32 feature_modules[] = {
        2u, 1u, residual_module, 2u, 1u, residual_module, 3u
    };
    const cellpack::u32 row_offsets[] = {
        0u, 2u, 4u, 6u, 8u
    };
    const cellpack::u32 row_modules[] = {
        1u, 2u,
        2u, 1u,
        3u, residual_module,
        1u, 3u
    };

    cellpack::feature_module_assignment_view features;
    features.feature_to_module = feature_modules;
    features.feature_count = 7u;
    features.residual_module_id = residual_module;

    cellpack::row_signature_view rows;
    rows.row_count = 4u;
    rows.row_offsets = row_offsets;
    rows.module_ids = row_modules;
    rows.entry_count = 8u;

    cellpack::planner_config config;
    config.residual_module_id = residual_module;
    config.min_primary_rows = 2u;

    cellpack::static_plan plan;
    cellpack::validation_result result = cellpack::build_static_plan(features, rows, config, &plan);
    require(static_cast<bool>(result), result.message);

    require(plan.desc.version == cellpack::abi_version, "wrong ABI version");
    require(plan.desc.row_count == 4u && plan.desc.feature_count == 7u, "wrong plan dimensions");
    require(plan.modules.size() == 4u, "expected three primary modules plus residual module");
    require(plan.modules[0].module_id == 1u && plan.modules[0].feature_begin == 0u && plan.modules[0].feature_count == 2u, "module 1 descriptor mismatch");
    require(plan.modules[1].module_id == 2u && plan.modules[1].feature_begin == 2u && plan.modules[1].feature_count == 2u, "module 2 descriptor mismatch");
    require(plan.modules[2].module_id == 3u && plan.modules[2].feature_begin == 4u && plan.modules[2].feature_count == 1u, "module 3 descriptor mismatch");
    require((plan.modules[3].flags & cellpack::module_flag_residual) != 0u, "residual module flag missing");

    const cellpack::u32 expected_feature_perm[] = { 1u, 4u, 0u, 3u, 6u, 2u, 5u };
    for (cellpack::u32 i = 0; i < 7u; ++i) {
        require(plan.feature_permutation[i] == expected_feature_perm[i], "feature permutation mismatch");
        require(plan.feature_permutation[plan.inverse_feature_permutation[i]] == i, "feature inverse mismatch");
    }

    require(plan.row_groups.size() == 3u, "expected three row-signature groups");
    require(plan.row_permutation[0] == 0u && plan.row_permutation[1] == 1u, "first row group should contain equivalent module signatures");
    for (cellpack::u32 i = 0; i < 4u; ++i) {
        require(plan.row_permutation[plan.inverse_row_permutation[i]] == i, "row inverse mismatch");
    }

    const cellpack::packed_region_desc *module_one = find_region(plan, 1u, cellpack::region_role::primary);
    const cellpack::packed_region_desc *module_two = find_region(plan, 2u, cellpack::region_role::primary);
    const cellpack::packed_region_desc *residual = find_region(plan, residual_module, cellpack::region_role::residual);
    require(module_one != nullptr && module_two != nullptr, "primary regions missing");
    require(module_one->row_begin == 0u && module_one->row_count == 2u, "module 1 primary row span mismatch");
    require(module_two->feature_begin == 2u && module_two->feature_count == 2u, "module 2 feature span mismatch");
    require(residual != nullptr, "residual region missing");
    require(residual->row_begin == 0u && residual->row_count == 4u, "residual row span mismatch");
    require(residual->feature_begin == 5u && residual->feature_count == 2u, "residual feature span mismatch");
    require(plan.desc.residual_region_count == 1u, "residual region count mismatch");

    cellpack::static_plan second_plan;
    result = cellpack::build_static_plan(features, rows, config, &second_plan);
    require(static_cast<bool>(result), result.message);
    require(plan.row_permutation == second_plan.row_permutation, "row permutation is not deterministic");
    require(plan.feature_permutation == second_plan.feature_permutation, "feature permutation is not deterministic");
    require(plan.regions.size() == second_plan.regions.size(), "region count is not deterministic");
    for (std::size_t i = 0; i < plan.regions.size(); ++i) {
        require(plan.regions[i].module_id == second_plan.regions[i].module_id, "region module order is not deterministic");
        require(plan.regions[i].row_begin == second_plan.regions[i].row_begin, "region row order is not deterministic");
        require(plan.regions[i].feature_begin == second_plan.regions[i].feature_begin, "region feature order is not deterministic");
    }

    const cellpack::u32 bad_row_modules[] = { 42u };
    const cellpack::u32 bad_offsets[] = { 0u, 1u };
    cellpack::row_signature_view bad_rows;
    bad_rows.row_count = 1u;
    bad_rows.row_offsets = bad_offsets;
    bad_rows.module_ids = bad_row_modules;
    bad_rows.entry_count = 1u;
    result = cellpack::build_static_plan(features, bad_rows, config, &second_plan);
    require(result.code == cellpack::validation_code::unknown_module, "unknown row-signature module was accepted");

    return 0;
}
