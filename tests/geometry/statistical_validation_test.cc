#include <Cellerator/geometry/statistical_validation.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(double actual, double expected, const char *message) {
    if (std::fabs(actual - expected) > 1.0e-12) throw std::runtime_error(message);
}

void require_code(
    const cellpack::validation_result &status,
    cellpack::validation_code expected,
    const char *message) {
    if (status.code != expected) throw std::runtime_error(message);
}

void test_metric_schema_preserves_denominators() {
    cellpack::packing_validation_metrics metrics;
    metrics.available = cellpack::packing_validation_metric_storage
        | cellpack::packing_validation_metric_records
        | cellpack::packing_validation_metric_tiles
        | cellpack::packing_validation_metric_preprocessing
        | cellpack::packing_validation_metric_runtime
        | cellpack::packing_validation_metric_correctness;
    metrics.dataset_identity = 11u;
    metrics.feature_axis_identity = 12u;
    metrics.row_domain_identity = 13u;
    metrics.split_identity = 14u;
    metrics.row_count = 5u;
    metrics.feature_count = 7u;
    metrics.nnz_count = 20u;
    metrics.encoded_bytes = 80u;
    metrics.metadata_bytes = 20u;
    metrics.baseline_bytes = 160u;
    metrics.active_block_references = 15u;
    metrics.tile_count = 4u;
    metrics.tile_block_union_references = 12u;
    metrics.padding_slots = 10u;
    metrics.preprocessing_input_nnz = 100u;
    metrics.preprocessing_elapsed_nanoseconds = 50u;
    metrics.runtime_input_nnz = 200u;
    metrics.runtime_bytes = 400u;
    metrics.runtime_elapsed_nanoseconds = 100u;
    metrics.correctness_items = 20u;
    metrics.preprocessing_repeat_count = 3u;
    metrics.runtime_repeat_count = 7u;

    cellpack::packing_validation_metric_rates rates;
    cellpack::validation_result status =
        cellpack::derive_packing_validation_metric_rates(metrics, &rates);
    require(static_cast<bool>(status), status.message);
    require_close(rates.encoded_bytes_per_nnz, 4.0, "encoded bytes/NNZ mismatch");
    require_close(rates.metadata_bytes_per_nnz, 1.0, "metadata bytes/NNZ mismatch");
    require_close(rates.compression_ratio, 2.0, "compression ratio mismatch");
    require_close(rates.active_blocks_per_row, 3.0, "blocks/row mismatch");
    require_close(rates.tile_block_union_per_tile, 3.0, "tile union/tile mismatch");
    require_close(rates.padding_slots_per_nnz, 0.5, "padding/NNZ mismatch");
    require_close(rates.preprocessing_nnz_per_second, 2.0e9,
        "preprocessing throughput mismatch");
    require_close(rates.runtime_nnz_per_second, 2.0e9, "runtime throughput mismatch");
    require_close(rates.runtime_gigabytes_per_second, 4.0, "runtime bandwidth mismatch");
    require(rates.exact_correctness, "zero mismatches must report exact correctness");

    metrics.runtime_repeat_count = 0u;
    require_code(cellpack::validate_packing_validation_metrics(metrics),
        cellpack::validation_code::invalid_plan_geometry,
        "runtime metrics without repeats were accepted");
    metrics.runtime_repeat_count = 7u;
    metrics.available |= 1u << 31u;
    require_code(cellpack::validate_packing_validation_metrics(metrics),
        cellpack::validation_code::unsupported_version,
        "unknown metric availability flag was accepted");
}

void test_group_aware_split_and_leakage_detection() {
    const cellpack::u64 rows[] = {101u, 102u, 103u, 104u, 105u, 106u};
    const cellpack::u64 groups[] = {7u, 7u, 8u, 9u, 9u, 10u};
    const cellpack::validation_identity_view identities{6u, rows, groups};
    std::vector<cellpack::validation_partition> first(6u), second(6u);
    cellpack::validation_split_provenance first_provenance, second_provenance;
    const cellpack::validation_split_config config{0x1234u, 2u};
    cellpack::validation_result status = cellpack::build_validation_split(
        identities, config, {first.size(), first.data()}, &first_provenance);
    require(static_cast<bool>(status), status.message);
    status = cellpack::build_validation_split(
        identities, config, {second.size(), second.data()}, &second_provenance);
    require(static_cast<bool>(status), status.message);
    require(first == second, "group-aware split is not deterministic");
    require(first_provenance.assignment_identity == second_provenance.assignment_identity,
        "deterministic split identity mismatch");
    require(first[0] == first[1] && first[3] == first[4],
        "caller groups crossed split partitions");
    require(first_provenance.claims_group_generalization,
        "group-aware split did not record group-generalization semantics");
    status = cellpack::validate_validation_split(identities, first.data(), first_provenance);
    require(static_cast<bool>(status), status.message);

    std::vector<cellpack::validation_partition> leaked = first;
    leaked[1] = leaked[0] == cellpack::validation_partition::training
        ? cellpack::validation_partition::held_out
        : cellpack::validation_partition::training;
    require_code(cellpack::validate_validation_split(
        identities, leaked.data(), first_provenance),
        cellpack::validation_code::invalid_permutation,
        "group leakage was not rejected");

    const cellpack::validation_identity_view cell_level{6u, rows, nullptr};
    cellpack::validation_split_provenance cell_provenance;
    status = cellpack::build_validation_split(
        cell_level, config, {second.size(), second.data()}, &cell_provenance);
    require(static_cast<bool>(status), status.message);
    require(!cell_provenance.claims_group_generalization
        && cell_provenance.unit_kind == cellpack::validation_unit_kind::row_identity,
        "cell-level split overclaimed group generalization");

    const cellpack::u64 duplicate_rows[] = {1u, 1u};
    const cellpack::validation_identity_view duplicates{2u, duplicate_rows, nullptr};
    std::vector<cellpack::validation_partition> duplicate_output(2u);
    require_code(cellpack::build_validation_split(
        duplicates, {1u, 1u}, {duplicate_output.size(), duplicate_output.data()},
        &cell_provenance), cellpack::validation_code::duplicate_id,
        "duplicate canonical row identities were accepted");
}

void test_group_aware_bootstrap_provenance() {
    const cellpack::u64 rows[] = {201u, 202u, 203u, 204u, 205u};
    const cellpack::u64 groups[] = {20u, 20u, 21u, 22u, 22u};
    const cellpack::validation_identity_view identities{5u, rows, groups};
    std::vector<cellpack::u32> first(5u), second(5u);
    cellpack::validation_bootstrap_provenance first_provenance, second_provenance;
    const cellpack::validation_bootstrap_config config{0xabcdu, 12u};
    cellpack::validation_result status = cellpack::build_validation_bootstrap(
        identities, config, {first.size(), first.data()}, &first_provenance);
    require(static_cast<bool>(status), status.message);
    status = cellpack::build_validation_bootstrap(
        identities, config, {second.size(), second.data()}, &second_provenance);
    require(static_cast<bool>(status), status.message);
    require(first == second, "group-aware bootstrap is not deterministic");
    require(first[0] == first[1] && first[3] == first[4],
        "caller group received inconsistent bootstrap multiplicity");
    require(first_provenance.bootstrap_identity == second_provenance.bootstrap_identity,
        "deterministic bootstrap identity mismatch");
    status = cellpack::validate_validation_bootstrap(
        identities, first.data(), first_provenance);
    require(static_cast<bool>(status), status.message);

    std::vector<cellpack::u32> tampered = first;
    ++tampered[1];
    require_code(cellpack::validate_validation_bootstrap(
        identities, tampered.data(), first_provenance),
        cellpack::validation_code::invalid_permutation,
        "inconsistent group bootstrap multiplicity was accepted");
}

struct null_fixture {
    std::vector<cellpack::u32> offsets;
    std::vector<cellpack::u32> features;
    cellpack::u32 feature_count = 0u;

    cellpack::csr_support_view view() const {
        return {static_cast<cellpack::u32>(offsets.size() - 1u), feature_count,
            static_cast<cellpack::u32>(features.size()), offsets.data(),
            features.empty() ? nullptr : features.data()};
    }
};

void test_exact_degree_preserving_null_reference() {
    const null_fixture source{{0u, 2u, 4u, 6u, 8u},
        {0u, 1u, 1u, 2u, 2u, 3u, 0u, 3u}, 4u};
    const std::vector<cellpack::u32> source_features = source.features;
    std::vector<cellpack::u32> offsets_a(5u), features_a(8u);
    std::vector<cellpack::u32> offsets_b(5u), features_b(8u);
    cellpack::csr_support_view output_a, output_b;
    cellpack::degree_preserving_null_provenance provenance_a, provenance_b;
    const cellpack::degree_preserving_null_config config{
        0x987654321u, 0x44415441534554u, 8u, 4096u};
    cellpack::validation_result status = cellpack::build_degree_preserving_null_reference(
        source.view(), config, {offsets_a.size(), features_a.size(),
            offsets_a.data(), features_a.data()}, &output_a, &provenance_a);
    require(static_cast<bool>(status), status.message);
    status = cellpack::build_degree_preserving_null_reference(
        source.view(), config, {offsets_b.size(), features_b.size(),
            offsets_b.data(), features_b.data()}, &output_b, &provenance_b);
    require(static_cast<bool>(status), status.message);
    require(offsets_a == offsets_b && features_a == features_b,
        "degree-preserving null output is not deterministic");
    require(provenance_a.output_identity == provenance_b.output_identity,
        "degree-preserving null identity is not deterministic");
    require(source.features == source_features, "null construction mutated source support");
    require(provenance_a.target_reached && provenance_a.accepted_swaps == config.requested_swaps,
        "mixable null fixture did not reach requested swap count");
    require(features_a != source.features, "null reference did not alter mixable support");

    cellpack::degree_conservation_report conservation;
    status = cellpack::validate_degree_conservation(source.view(), output_a, &conservation);
    require(static_cast<bool>(status), status.message);
    require(conservation.exact && conservation.row_degree_mismatches == 0u
        && conservation.feature_degree_mismatches == 0u,
        "null reference did not conserve exact row/feature degrees");
    status = cellpack::validate_degree_preserving_null_provenance(
        source.view(), output_a, provenance_a);
    require(static_cast<bool>(status), status.message);

    std::vector<cellpack::u32> tampered_features = features_a;
    const cellpack::u32 old_feature = tampered_features[0];
    tampered_features[0] = old_feature == 0u ? 1u : 0u;
    std::sort(tampered_features.begin(), tampered_features.begin() + 2u);
    const cellpack::csr_support_view tampered{4u, 4u, 8u,
        offsets_a.data(), tampered_features.data()};
    status = cellpack::validate_degree_preserving_null_provenance(
        source.view(), tampered, provenance_a);
    require(!static_cast<bool>(status), "tampered null output retained valid provenance");
}

void test_null_reports_unmixable_support_without_relaxing_degrees() {
    const null_fixture complete{{0u, 2u, 4u}, {0u, 1u, 0u, 1u}, 2u};
    std::vector<cellpack::u32> offsets(3u), features(4u);
    cellpack::csr_support_view output;
    cellpack::degree_preserving_null_provenance provenance;
    const cellpack::degree_preserving_null_config config{17u, 18u, 3u, 64u};
    cellpack::validation_result status = cellpack::build_degree_preserving_null_reference(
        complete.view(), config, {offsets.size(), features.size(), offsets.data(), features.data()},
        &output, &provenance);
    require(static_cast<bool>(status), status.message);
    require(!provenance.target_reached && provenance.accepted_swaps == 0u
        && provenance.attempted_swaps == config.maximum_attempts,
        "unmixable null did not expose failed mixing target");
    require(features == complete.features, "unmixable null changed complete support");
    status = cellpack::validate_degree_preserving_null_provenance(
        complete.view(), output, provenance);
    require(static_cast<bool>(status), status.message);
}

} // namespace

int main() {
    test_metric_schema_preserves_denominators();
    test_group_aware_split_and_leakage_detection();
    test_group_aware_bootstrap_provenance();
    test_exact_degree_preserving_null_reference();
    test_null_reports_unmixable_support_without_relaxing_degrees();
    return 0;
}
