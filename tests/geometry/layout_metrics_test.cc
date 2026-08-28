#include <Cellerator/geometry/layout_metrics.hh>

#include <cmath>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(double actual, double expected, const char *message) {
    if (std::fabs(actual - expected) > 1.0e-9) throw std::runtime_error(message);
}

cellpack::static_plan build_fixture_plan() {
    const cellpack::u32 feature_modules[] = {
        1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u
    };
    const cellpack::u32 row_offsets[] = {
        0u, 2u, 4u, 6u, 8u
    };
    const cellpack::u32 row_modules[] = {
        1u, 2u,
        1u, 2u,
        1u, 2u,
        1u, 2u
    };

    cellpack::feature_module_assignment_view features;
    features.feature_to_module = feature_modules;
    features.feature_count = 8u;

    cellpack::row_signature_view rows;
    rows.row_count = 4u;
    rows.row_offsets = row_offsets;
    rows.module_ids = row_modules;
    rows.entry_count = 8u;

    cellpack::static_plan plan;
    cellpack::validation_result result = cellpack::build_static_plan(features, rows, cellpack::planner_config{}, &plan);
    require(static_cast<bool>(result), result.message);
    require(plan.regions.size() == 2u, "fixture should produce two module regions");
    return plan;
}

const cellpack::region_layout_metrics *find_metrics(
    const cellpack::layout_metrics_plan &metrics,
    cellpack::u32 region_id) {
    for (const cellpack::region_layout_metrics &region : metrics.regions) {
        if (region.region_id == region_id) return &region;
    }
    return nullptr;
}

} // namespace

int main() {
    const cellpack::u32 csr_offsets[] = { 0u, 4u, 7u, 14u, 16u };
    const cellpack::u32 csr_features[] = {
        0u, 1u, 4u, 5u,
        0u, 2u, 4u,
        1u, 2u, 3u, 4u, 5u, 6u, 7u,
        0u, 4u
    };
    const float csr_values[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
        15.0f, 16.0f
    };

    cellpack::static_plan plan = build_fixture_plan();
    cellpack::csr_view csr;
    csr.row_count = 4u;
    csr.feature_count = 8u;
    csr.nnz_count = 16u;
    csr.row_offsets = csr_offsets;
    csr.feature_ids = csr_features;
    csr.values = csr_values;

    cellpack::packed_coordinate_plan packed;
    cellpack::validation_result result = cellpack::build_packed_coordinate_plan(csr, plan, &packed);
    require(static_cast<bool>(result), result.message);

    cellpack::layout_metrics_config config;
    config.sliced_ell_slice_height = 2u;
    cellpack::layout_metrics_plan metrics;
    result = cellpack::build_layout_metrics(plan, packed, config, &metrics);
    require(static_cast<bool>(result), result.message);
    require(metrics.nnz == 16u, "plan nnz mismatch");
    require(metrics.regions.size() == 2u, "metrics region count mismatch");

    const cellpack::region_layout_metrics *module_one = find_metrics(metrics, 0u);
    const cellpack::region_layout_metrics *module_two = find_metrics(metrics, 1u);
    require(module_one != nullptr && module_two != nullptr, "module metrics missing");

    require(module_one->row_count == 4u && module_one->feature_count == 4u, "module one shape mismatch");
    require(module_one->nnz == 8u, "module one nnz mismatch");
    require(module_one->row_widths.min_width == 1u, "module one min width mismatch");
    require(module_one->row_widths.max_width == 3u, "module one max width mismatch");
    require(module_one->row_widths.width_sum == 8u, "module one width sum mismatch");
    require(module_one->row_widths.squared_width_sum == 18u, "module one squared width sum mismatch");
    require(module_one->blocked_ell_padded_slots == 12u, "module one Blocked-ELL padded slots mismatch");
    require(module_one->sliced_ell_padded_slots == 10u, "module one Sliced-ELL padded slots mismatch");
    require(module_one->dense_tile_slots == 16u, "module one dense slots mismatch");
    require_close(module_one->blocked_ell_fill_ratio, 8.0 / 12.0, "module one Blocked-ELL fill mismatch");
    require_close(module_one->sliced_ell_fill_ratio, 8.0 / 10.0, "module one Sliced-ELL fill mismatch");
    require_close(module_one->dense_tile_fill_ratio, 0.5, "module one dense fill mismatch");
    require(module_one->csr_index_bytes == 52u, "module one CSR index bytes mismatch");
    require(module_one->csr_value_bytes == 32u, "module one CSR value bytes mismatch");
    require(module_one->blocked_ell_index_bytes == 48u, "module one Blocked-ELL index bytes mismatch");
    require(module_one->blocked_ell_value_bytes == 48u, "module one Blocked-ELL value bytes mismatch");
    require(module_one->sliced_ell_index_bytes == 40u, "module one Sliced-ELL index bytes mismatch");
    require(module_one->sliced_ell_value_bytes == 40u, "module one Sliced-ELL value bytes mismatch");
    require(module_one->estimated_output_bytes == 16u, "module one output bytes mismatch");
    require_close(module_one->residual_fraction, 0.5, "module one residual fraction mismatch");

    require(module_two->nnz == 8u, "module two nnz mismatch");
    require(module_two->blocked_ell_padded_slots == 16u, "module two Blocked-ELL padded slots mismatch");
    require(module_two->sliced_ell_padded_slots == 12u, "module two Sliced-ELL padded slots mismatch");

    cellpack::layout_plan_summary summary = cellpack::summarize_layout_metrics(metrics);
    require(summary.region_count == 2u, "summary region count mismatch");
    require(summary.launch_group_count == 2u, "summary launch group count mismatch");
    require(summary.total_estimated_bytes == cellpack::csr_estimated_bytes(*module_one) + cellpack::csr_estimated_bytes(*module_two),
            "summary estimated bytes mismatch");
    require_close(summary.mean_blocked_ell_fill, ((8.0 / 12.0) + 0.5) / 2.0, "summary mean fill mismatch");

    cellpack::layout_metrics_config bad_config;
    bad_config.value_bytes = 0u;
    result = cellpack::build_layout_metrics(plan, packed, bad_config, &metrics);
    require(result.code == cellpack::validation_code::invalid_matrix_view, "invalid metrics config was accepted");

    return 0;
}
