#include <Cellerator/geometry/pack.hh>

#include <cstddef>
#include <cmath>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

bool same_value(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 0.0f;
}

const cellpack::packed_region_desc *find_region(const cellpack::static_plan &plan, cellpack::u32 region_id) {
    for (const cellpack::packed_region_desc &region : plan.regions) {
        if (region.region_id == region_id) return &region;
    }
    return nullptr;
}

cellpack::static_plan build_fixture_plan() {
    constexpr cellpack::u32 residual_module = 99u;
    const cellpack::u32 feature_modules[] = {
        2u, 1u, residual_module, 2u, 1u, residual_module, 3u
    };
    const cellpack::u32 row_offsets[] = {
        0u, 2u, 4u, 6u, 8u, 10u
    };
    const cellpack::u32 row_modules[] = {
        1u, 2u,
        2u, 1u,
        3u, residual_module,
        1u, 3u,
        1u, 2u
    };

    cellpack::feature_module_assignment_view features;
    features.feature_to_module = feature_modules;
    features.feature_count = 7u;
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
    require((plan.desc.row_permutation.flags & cellpack::permutation_flag_identity) == 0u, "row permutation should be non-identity");
    require((plan.desc.feature_permutation.flags & cellpack::permutation_flag_identity) == 0u, "feature permutation should be non-identity");
    return plan;
}

void require_same_csr(
    const cellpack::reconstructed_csr &actual,
    const cellpack::u32 *expected_offsets,
    const cellpack::u32 *expected_features,
    const float *expected_values,
    cellpack::u32 row_count,
    cellpack::u32 feature_count,
    cellpack::u32 nnz_count) {
    require(actual.row_count == row_count, "reconstructed row count mismatch");
    require(actual.feature_count == feature_count, "reconstructed feature count mismatch");
    require(actual.row_offsets.size() == static_cast<std::size_t>(row_count) + 1u, "reconstructed row offset size mismatch");
    require(actual.feature_ids.size() == nnz_count, "reconstructed feature size mismatch");
    require(actual.values.size() == nnz_count, "reconstructed value size mismatch");
    for (cellpack::u32 i = 0; i <= row_count; ++i) {
        require(actual.row_offsets[i] == expected_offsets[i], "reconstructed row offsets mismatch");
    }
    for (cellpack::u32 i = 0; i < nnz_count; ++i) {
        require(actual.feature_ids[i] == expected_features[i], "reconstructed feature ids mismatch");
        require(same_value(actual.values[i], expected_values[i]), "reconstructed values mismatch");
    }
}

} // namespace

int main() {
    constexpr cellpack::u32 row_count = 5u;
    constexpr cellpack::u32 feature_count = 7u;
    constexpr cellpack::u32 nnz_count = 8u;
    const cellpack::u32 csr_offsets[] = { 0u, 2u, 2u, 4u, 6u, 8u };
    const cellpack::u32 csr_features[] = { 0u, 1u, 2u, 6u, 1u, 6u, 3u, 4u };
    const float csr_values[] = { 1.0f, 2.0f, 20.0f, 21.0f, 30.0f, 31.0f, 40.0f, 41.0f };

    cellpack::static_plan plan = build_fixture_plan();

    cellpack::csr_view csr;
    csr.row_count = row_count;
    csr.feature_count = feature_count;
    csr.nnz_count = nnz_count;
    csr.row_offsets = csr_offsets;
    csr.feature_ids = csr_features;
    csr.values = csr_values;

    cellpack::packed_coordinate_plan packed;
    cellpack::validation_result result = cellpack::build_packed_coordinate_plan(csr, plan, &packed);
    require(static_cast<bool>(result), result.message);
    require(packed.coordinates.size() == nnz_count, "CSR packed coordinate count mismatch");

    bool saw_residual = false;
    bool saw_permuted_coordinate = false;
    for (const cellpack::packed_coordinate &coordinate : packed.coordinates) {
        if (coordinate.original_feature == 2u) {
            const cellpack::packed_region_desc *region = find_region(plan, coordinate.region_id);
            require(region != nullptr, "residual coordinate region not found");
            require(region->role == cellpack::to_u32(cellpack::region_role::residual), "residual feature did not map to residual region");
            saw_residual = true;
        }
        if (coordinate.original_row != coordinate.permuted_row || coordinate.original_feature != coordinate.permuted_feature) {
            saw_permuted_coordinate = true;
        }
    }
    require(saw_residual, "residual feature did not produce a packed coordinate");
    require(saw_permuted_coordinate, "packed coordinates did not record non-identity permutation");

    cellpack::reconstructed_csr reconstructed;
    result = cellpack::reconstruct_csr_from_coordinate_plan(row_count, feature_count, plan, packed, &reconstructed);
    require(static_cast<bool>(result), result.message);
    require_same_csr(reconstructed, csr_offsets, csr_features, csr_values, row_count, feature_count, nnz_count);

    const cellpack::u32 coo_rows[] = { 0u, 0u, 2u, 2u, 3u, 3u, 4u, 4u };
    cellpack::coo_view coo;
    coo.row_count = row_count;
    coo.feature_count = feature_count;
    coo.nnz_count = nnz_count;
    coo.row_ids = coo_rows;
    coo.feature_ids = csr_features;
    coo.values = csr_values;

    cellpack::packed_coordinate_plan coo_packed;
    result = cellpack::build_packed_coordinate_plan(coo, plan, &coo_packed);
    require(static_cast<bool>(result), result.message);
    require(coo_packed.coordinates.size() == nnz_count, "COO packed coordinate count mismatch");

    cellpack::reconstructed_csr coo_reconstructed;
    result = cellpack::reconstruct_csr_from_coordinate_plan(row_count, feature_count, plan, coo_packed, &coo_reconstructed);
    require(static_cast<bool>(result), result.message);
    require_same_csr(coo_reconstructed, csr_offsets, csr_features, csr_values, row_count, feature_count, nnz_count);

    const cellpack::u32 bad_offsets[] = { 0u, 1u };
    const cellpack::u32 bad_features[] = { 6u };
    const float bad_values[] = { 1.0f };
    cellpack::feature_module_assignment_view bad_feature_modules;
    const cellpack::u32 bad_feature_to_module[] = { 1u, 1u, 1u, 1u, 1u, 1u, 3u };
    bad_feature_modules.feature_to_module = bad_feature_to_module;
    bad_feature_modules.feature_count = feature_count;

    const cellpack::u32 bad_signature_offsets[] = { 0u, 1u };
    const cellpack::u32 bad_signature_modules[] = { 1u };
    cellpack::row_signature_view bad_rows;
    bad_rows.row_count = 1u;
    bad_rows.row_offsets = bad_signature_offsets;
    bad_rows.module_ids = bad_signature_modules;
    bad_rows.entry_count = 1u;
    cellpack::static_plan missing_region_plan;
    result = cellpack::build_static_plan(bad_feature_modules, bad_rows, cellpack::planner_config{}, &missing_region_plan);
    require(static_cast<bool>(result), result.message);

    cellpack::csr_view bad_csr;
    bad_csr.row_count = 1u;
    bad_csr.feature_count = feature_count;
    bad_csr.nnz_count = 1u;
    bad_csr.row_offsets = bad_offsets;
    bad_csr.feature_ids = bad_features;
    bad_csr.values = bad_values;
    result = cellpack::build_packed_coordinate_plan(bad_csr, missing_region_plan, &packed);
    require(result.code == cellpack::validation_code::missing_region, "source entry without a precompiled region was accepted");

    return 0;
}
