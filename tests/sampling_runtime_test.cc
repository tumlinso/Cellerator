#include <Cellerator/compute/sampling.hh>

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace cs = cellerator::compute::sampling;
namespace cm = cellerator::matrix;
namespace ct = cellerator::types;

int require(bool ok, const char *label) {
    if (!ok) std::fprintf(stderr, "%s\n", label);
    return ok ? 1 : 0;
}

bool close_f64(double left, double right) {
    return std::fabs(left - right) < 1.0e-12;
}

bool fill_source(cm::compressed *matrix) {
    cm::init(matrix, 8u, 5u, 14u, cm::compressed_by_row);
    if (!cm::allocate(matrix)) return false;
    const ct::ptr_t ptr[] = {0u, 2u, 3u, 6u, 8u, 9u, 11u, 12u, 14u};
    const ct::idx_t idx[] = {0u, 4u, 2u, 0u, 1u, 3u, 1u, 4u, 2u, 0u, 3u, 4u, 1u, 2u};
    for (std::size_t i = 0u; i < 9u; ++i) matrix->majorPtr[i] = ptr[i];
    for (std::size_t i = 0u; i < 14u; ++i) {
        matrix->minorIdx[i] = idx[i];
        matrix->val[i] = __float2half((float) i + 1.0f);
    }
    return true;
}

int test_golden_hashes() {
    if (!require(cs::splitmix64_hash(0u) == 0xe220a8397b1dcdafull, "SplitMix64 golden zero mismatch")) return 1;
    if (!require(cs::splitmix64_hash(1u) == 0x910a2dec89025cc1ull, "SplitMix64 golden one mismatch")) return 2;
    if (!require(cs::splitmix64_hash(UINT64_MAX) == 0xe4d971771b652c20ull, "SplitMix64 golden max mismatch")) return 3;
    if (!require(cs::hash_global_row_index(42u, 7u) == 0x0dcbfdc070e13accull,
                 "global-row hash golden mismatch")) return 4;
    const char barcode[] = "AAACCCAAGAAACCAT-1";
    if (!require(cs::hash_stable_cell_id(barcode, sizeof(barcode) - 1u, 17u) == 0x7f9677d97d736275ull,
                 "stable-cell-ID hash golden mismatch")) return 5;
    return 0;
}

int test_quantile_splits_are_deterministic_and_disjoint() {
    constexpr std::uint64_t rows = 100000u;
    cs::sample_spec train, validation, test;
    cs::sample_plan train_a, train_b, validation_plan, test_plan, replay;
    cs::cell_identity_view identities;
    std::string error;
    train.seed = validation.seed = test.seed = 41u;
    train.split_name = "train";
    validation.split_name = "validation";
    test.split_name = "test";
    train.quantile = {{0u, 1000u}, {24u, 1000u}};
    validation.quantile = {{24u, 1000u}, {30u, 1000u}};
    test.quantile = {{30u, 1000u}, {36u, 1000u}};

    if (!require(cs::build_sample_plan(rows, train, identities, &train_a, &error), error.c_str())) return 10;
    if (!require(cs::build_sample_plan(rows, train, identities, &train_b, &error), error.c_str())) return 11;
    if (!require(cs::build_sample_plan(rows, validation, identities, &validation_plan, &error), error.c_str())) return 12;
    if (!require(cs::build_sample_plan(rows, test, identities, &test_plan, &error), error.c_str())) return 13;
    if (!require(train_a.global_row_indices == train_b.global_row_indices
                 && train_a.identity_hashes == train_b.identity_hashes,
                 "quantile sampling is not deterministic")) return 14;
    if (!require(!train_a.global_row_indices.empty()
                 && !validation_plan.global_row_indices.empty()
                 && !test_plan.global_row_indices.empty(),
                 "quantile split fixture unexpectedly selected no rows")) return 15;
    if (!require(std::is_sorted(train_a.global_row_indices.begin(), train_a.global_row_indices.end()),
                 "quantile sample rows are not ascending")) return 16;

    std::vector<std::uint8_t> membership((std::size_t) rows, 0u);
    for (std::uint64_t row : train_a.global_row_indices) membership[(std::size_t) row] |= 1u;
    for (std::uint64_t row : validation_plan.global_row_indices) membership[(std::size_t) row] |= 2u;
    for (std::uint64_t row : test_plan.global_row_indices) membership[(std::size_t) row] |= 4u;
    for (std::uint8_t value : membership) {
        if (!require(value == 0u || value == 1u || value == 2u || value == 4u,
                     "quantile split ranges overlap")) return 17;
    }
    if (!require(train_a.provenance.seed == 41u
                 && train_a.provenance.hash_algorithm == cs::splitmix64_algorithm_name
                 && train_a.provenance.hash_version == cs::splitmix64_algorithm_version
                 && train_a.provenance.total_rows == rows
                 && train_a.provenance.selected_rows == train_a.global_row_indices.size()
                 && train_a.provenance.mode == cs::selection_mode::hash_quantile_range
                 && train_a.provenance.split_name == "train"
                 && train_a.provenance.cell_identity == cs::cell_identity_kind::global_row_index,
                 "quantile provenance is incomplete")) return 18;
    if (!require(cs::reproduce_sample_plan(train_a.provenance, identities, &replay, &error), error.c_str())) return 19;
    if (!require(replay.global_row_indices == train_a.global_row_indices
                 && replay.identity_hashes == train_a.identity_hashes,
                 "quantile sample did not replay from provenance")) return 20;
    return 0;
}

int test_exact_size_and_stable_cell_ids() {
    const char *ids[] = {"cell-g", "cell-c", "cell-a", "cell-h", "cell-b", "cell-f", "cell-d", "cell-e"};
    const std::size_t permutation[] = {2u, 7u, 0u, 4u, 6u, 1u, 5u, 3u};
    const char *reordered_ids[8];
    cs::sample_spec spec;
    cs::sample_plan original, reordered, replay;
    cs::cell_identity_view original_view, reordered_view;
    std::string error;
    for (std::size_t i = 0u; i < 8u; ++i) reordered_ids[i] = ids[permutation[i]];
    spec.mode = cs::selection_mode::exact_lowest_hash;
    spec.seed = 99u;
    spec.split_name = "exact-three";
    spec.requested_row_count = 3u;
    original_view.kind = reordered_view.kind = cs::cell_identity_kind::stable_item_id;
    original_view.stable_cell_ids = ids;
    reordered_view.stable_cell_ids = reordered_ids;
    original_view.count = reordered_view.count = 8u;

    if (!require(cs::build_sample_plan(8u, spec, original_view, &original, &error), error.c_str())) return 20;
    if (!require(cs::build_sample_plan(8u, spec, reordered_view, &reordered, &error), error.c_str())) return 21;
    if (!require(original.global_row_indices.size() == 3u
                 && reordered.global_row_indices.size() == 3u,
                 "exact sample size mismatch")) return 22;
    if (!require(std::is_sorted(original.global_row_indices.begin(), original.global_row_indices.end())
                 && std::is_sorted(reordered.global_row_indices.begin(), reordered.global_row_indices.end()),
                 "exact sample rows are not ascending")) return 23;

    std::vector<std::uint64_t> all_hashes;
    for (const char *id : ids) all_hashes.push_back(cs::hash_stable_cell_id(id, std::strlen(id), spec.seed));
    std::sort(all_hashes.begin(), all_hashes.end());
    all_hashes.resize(3u);
    std::vector<std::uint64_t> selected_hashes = original.identity_hashes;
    std::sort(selected_hashes.begin(), selected_hashes.end());
    if (!require(selected_hashes == all_hashes, "exact sample did not select the lowest identity hashes")) return 24;

    std::vector<std::string> original_selected, reordered_selected;
    for (std::uint64_t row : original.global_row_indices) original_selected.emplace_back(ids[(std::size_t) row]);
    for (std::uint64_t row : reordered.global_row_indices) reordered_selected.emplace_back(reordered_ids[(std::size_t) row]);
    std::sort(original_selected.begin(), original_selected.end());
    std::sort(reordered_selected.begin(), reordered_selected.end());
    if (!require(original_selected == reordered_selected,
                 "stable item ID selection depends on physical row order")) return 25;
    if (!require(original.provenance.cell_identity == cs::cell_identity_kind::stable_item_id
                 && original.provenance.requested_row_count == 3u,
                 "stable item ID provenance mismatch")) return 26;
    if (!require(cs::reproduce_sample_plan(original.provenance, original_view, &replay, &error), error.c_str())) return 27;
    if (!require(replay.global_row_indices == original.global_row_indices,
                 "stable-ID exact sample did not replay from provenance")) return 28;
    return 0;
}

int test_exact_global_rows_limits_and_small_population() {
    constexpr std::uint64_t rows = 70000u;
    cs::sample_spec spec;
    cs::sample_plan first, second, rejected, small, replay;
    cs::cell_identity_view identities;
    std::string error;
    spec.mode = cs::selection_mode::exact_lowest_hash;
    spec.seed = 0x123456789abcdef0ull;
    spec.split_name = "exact-global-65536";
    spec.requested_row_count = cs::maximum_exact_sample_rows;

    if (!require(cs::build_sample_plan(rows, spec, identities, &first, &error), error.c_str())) return 70;
    if (!require(cs::build_sample_plan(rows, spec, identities, &second, &error), error.c_str())) return 71;
    if (!require(first.global_row_indices == second.global_row_indices
                 && first.identity_hashes == second.identity_hashes,
                 "maximum exact global-row sample positions are not deterministic")) return 72;
    if (!require(first.global_row_indices.size() == cs::maximum_exact_sample_rows
                 && first.identity_hashes.size() == first.global_row_indices.size()
                 && std::is_sorted(first.global_row_indices.begin(), first.global_row_indices.end()),
                 "maximum exact global-row sample shape or ordering mismatch")) return 73;

    std::vector<std::pair<std::uint64_t, std::uint64_t>> hash_order;
    hash_order.reserve((std::size_t) rows);
    for (std::uint64_t row = 0u; row < rows; ++row) {
        hash_order.emplace_back(cs::hash_global_row_index(row, spec.seed), row);
    }
    std::sort(hash_order.begin(), hash_order.end());
    hash_order.resize((std::size_t) cs::maximum_exact_sample_rows);
    std::vector<std::uint64_t> expected_rows;
    expected_rows.reserve(hash_order.size());
    for (const auto &entry : hash_order) expected_rows.push_back(entry.second);
    std::sort(expected_rows.begin(), expected_rows.end());
    if (!require(first.global_row_indices == expected_rows,
                 "maximum exact sample did not select the lowest global-row hashes")) return 74;
    for (std::size_t position = 0u; position < first.global_row_indices.size(); ++position) {
        if (!require(first.identity_hashes[position]
                         == cs::hash_global_row_index(first.global_row_indices[position], spec.seed),
                     "sample position lost its global-row/hash mapping")) return 75;
    }
    if (!require(first.provenance.seed == spec.seed
                 && first.provenance.hash_version == cs::splitmix64_algorithm_version
                 && first.provenance.total_rows == rows
                 && first.provenance.requested_row_count == cs::maximum_exact_sample_rows
                 && first.provenance.selected_rows == cs::maximum_exact_sample_rows
                 && first.provenance.split_name == spec.split_name
                 && first.provenance.cell_identity == cs::cell_identity_kind::global_row_index,
                 "maximum exact global-row provenance is incomplete")) return 76;

    spec.requested_row_count = cs::maximum_exact_sample_rows + 1u;
    error.clear();
    if (!require(!cs::build_sample_plan(rows, spec, identities, &rejected, &error)
                 && rejected.global_row_indices.empty(),
                 "exact sample accepted a request above its documented limit")) return 77;

    spec.split_name = "undersized-population";
    spec.requested_row_count = 64u;
    error.clear();
    if (!require(cs::build_sample_plan(7u, spec, identities, &small, &error), error.c_str())) return 78;
    const std::vector<std::uint64_t> all_rows = {0u, 1u, 2u, 3u, 4u, 5u, 6u};
    if (!require(small.global_row_indices == all_rows
                 && small.identity_hashes.size() == all_rows.size()
                 && small.provenance.total_rows == 7u
                 && small.provenance.requested_row_count == 64u
                 && small.provenance.selected_rows == 7u
                 && small.provenance.split_name == "undersized-population",
                 "undersized population was not selected completely with auditable provenance")) return 79;
    if (!require(cs::reproduce_sample_plan(small.provenance, identities, &replay, &error), error.c_str())) return 80;
    if (!require(replay.global_row_indices == small.global_row_indices
                 && replay.identity_hashes == small.identity_hashes,
                 "undersized exact sample did not replay from provenance")) return 81;
    return 0;
}

int test_density_stratified_csr_sampling() {
    cm::compressed source;
    cs::density_sample_spec spec;
    cs::cell_identity_view identities;
    cs::sample_plan first, second, generic, replay;
    const std::uint64_t row_nnz[] = {2u, 1u, 3u, 2u, 1u, 2u, 1u, 2u};
    std::string error;
    cm::init(&source);
    if (!require(fill_source(&source), "density source allocation failed")) return 50;
    spec.seed = 71u;
    spec.split_name = "density";
    spec.requested_strata = 3u;
    spec.requested_row_count = 5u;

    if (!require(cs::build_csr_density_sample_plan(&source, spec, identities, &first, &error), error.c_str())) return 51;
    if (!require(cs::build_csr_density_sample_plan(&source, spec, identities, &second, &error), error.c_str())) return 52;
    if (!require(first.global_row_indices == second.global_row_indices
                 && first.identity_hashes == second.identity_hashes
                 && first.row_strata == second.row_strata
                 && first.sampling_weights == second.sampling_weights,
                 "density sampling is not deterministic")) return 53;
    if (!require(first.global_row_indices.size() == 5u
                 && first.row_strata.size() == 5u
                 && first.sampling_weights.size() == 5u,
                 "density sample did not produce the exact requested size")) return 54;
    if (!require(std::is_sorted(first.global_row_indices.begin(), first.global_row_indices.end())
                 && std::adjacent_find(first.global_row_indices.begin(), first.global_row_indices.end())
                    == first.global_row_indices.end(),
                 "density sample rows are unsorted or duplicated")) return 55;

    const std::vector<std::uint64_t> expected_boundaries = {1u, 2u, 3u};
    const std::vector<std::uint64_t> expected_counts = {3u, 4u, 1u};
    const std::vector<std::uint64_t> expected_sampled = {2u, 2u, 1u};
    if (!require(first.provenance.mode == cs::selection_mode::density_quantile_exact
                 && first.provenance.density_strata == 3u
                 && first.provenance.density_bin_upper_bounds_inclusive == expected_boundaries
                 && first.provenance.stratum_total_rows == expected_counts
                 && first.provenance.stratum_sampled_rows == expected_sampled
                 && first.provenance.seed == spec.seed
                 && first.provenance.hash_version == cs::splitmix64_algorithm_version
                 && first.provenance.weighting_rule == cs::inverse_stratum_weighting_rule,
                 "density provenance or deterministic bins mismatch")) return 56;

    double weighted_constant_total = 0.0;
    for (std::size_t i = 0u; i < first.global_row_indices.size(); ++i) {
        const std::uint32_t stratum = first.row_strata[i];
        const double expected_weight = (double) expected_counts[stratum] / (double) expected_sampled[stratum];
        if (!require(close_f64(first.sampling_weights[i], expected_weight),
                     "density sample weight is not traceable to stratum counts")) return 57;
        weighted_constant_total += first.sampling_weights[i];
    }
    if (!require(close_f64(weighted_constant_total, (double) source.rows),
                 "density weights do not preserve the row-uniform constant objective")) return 58;

    const cs::row_nnz_view row_nnz_view{row_nnz, 8u};
    if (!require(cs::build_density_sample_plan(8u, row_nnz_view, spec, identities, &generic, &error), error.c_str())) return 59;
    if (!require(generic.global_row_indices == first.global_row_indices
                 && generic.sampling_weights == first.sampling_weights,
                 "generic structural row nnz path differs from CSR row pointers")) return 60;
    if (!require(cs::reproduce_density_sample_plan(first.provenance, row_nnz_view, identities, &replay, &error),
                 error.c_str())) return 61;
    if (!require(replay.global_row_indices == first.global_row_indices
                 && replay.row_strata == first.row_strata
                 && replay.sampling_weights == first.sampling_weights,
                 "density sample did not replay from provenance")) return 62;

    const std::uint64_t tied_row_nnz[] = {0u, 100u, 100u, 100u};
    cs::sample_plan tied_plan;
    cs::density_sample_spec tied_spec = spec;
    tied_spec.split_name = "density-ties";
    tied_spec.requested_strata = 2u;
    tied_spec.requested_row_count = 2u;
    if (!require(cs::build_density_sample_plan(4u, {tied_row_nnz, 4u}, tied_spec, identities, &tied_plan, &error),
                 error.c_str())) return 63;
    if (!require(tied_plan.provenance.density_bin_upper_bounds_inclusive == std::vector<std::uint64_t>({0u, 100u})
                 && tied_plan.provenance.stratum_total_rows == std::vector<std::uint64_t>({1u, 3u}),
                 "density quantiles collapsed a valid boundary around tied row lengths")) return 64;

    cs::density_sample_spec undersampled_spec = spec;
    cs::sample_plan rejected;
    undersampled_spec.split_name = "density-invalid";
    undersampled_spec.requested_row_count = 2u;
    error.clear();
    if (!require(!cs::build_density_sample_plan(8u, row_nnz_view, undersampled_spec, identities, &rejected, &error),
                 "density sampling allowed a non-empty stratum to be dropped")) return 65;

    cm::clear(&source);
    return 0;
}

int main() {
    int rc = test_golden_hashes();
    if (rc != 0) return rc;
    rc = test_quantile_splits_are_deterministic_and_disjoint();
    if (rc != 0) return rc;
    rc = test_exact_size_and_stable_cell_ids();
    if (rc != 0) return rc;
    rc = test_exact_global_rows_limits_and_small_population();
    if (rc != 0) return rc;
    return test_density_stratified_csr_sampling();
}
