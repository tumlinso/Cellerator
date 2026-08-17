#include <CellPack/record_statistical_validation.hh>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using cellpack::u32;
using cellpack::u64;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void require_status(cellpack::validation_result status, const char *message) {
    if (!status) throw std::runtime_error(message);
}

void require_code(
    cellpack::validation_result status,
    cellpack::validation_code expected,
    const char *message) {
    if (status.code != expected) throw std::runtime_error(message);
}

void require_close(double actual, double expected, const char *message) {
    if (std::fabs(actual - expected) > 1.0e-12) throw std::runtime_error(message);
}

cellpack::frozen_packing_plan make_plan() {
    const u32 feature_permutation[] = {0u, 1u, 2u, 3u, 4u, 5u};
    const u32 inverse_feature_permutation[] = {0u, 1u, 2u, 3u, 4u, 5u};
    const u32 feature_block_offsets[] = {0u, 3u, 6u};
    const u32 feature_to_block[] = {0u, 0u, 0u, 1u, 1u, 1u};
    const u32 feature_to_local[] = {0u, 1u, 2u, 0u, 1u, 2u};
    const u32 row_group_offsets[] = {0u, 4u, 8u};
    cellpack::frozen_packing_plan_build_view build;
    build.row_count = 8u;
    build.feature_count = 6u;
    build.feature_permutation = feature_permutation;
    build.inverse_feature_permutation = inverse_feature_permutation;
    build.feature_block_count = 2u;
    build.feature_block_offsets = feature_block_offsets;
    build.feature_to_block = feature_to_block;
    build.feature_to_local = feature_to_local;
    build.row_group_count = 2u;
    build.row_group_offsets = row_group_offsets;
    build.maximum_feature_block_width = 3u;
    build.row_group_width = 4u;
    build.identity.feature_axis_fingerprint = 0x4645415455524553ull;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind = cellpack::packing_row_domain_kind::full_dataset_identity;
    build.identity.row_domain_identity = 0x524f57444f4d4149ull;
    build.identity.evaluation_source_identity = 0x4556414c534f5552ull;
    build.cost_policy_identity = 0x434f5354504f4c49ull;
    cellpack::frozen_packing_plan plan;
    require_status(cellpack::freeze_packing_plan(build, &plan), "freeze plan");
    return plan;
}

struct canonical_matrix {
    std::vector<u32> row_offsets;
    std::vector<u32> feature_ids;
    std::vector<unsigned char> values;
    u64 dataset_identity = 0u;
    u64 global_row_begin = 0u;
    u32 full_row_count = 8u;

    cellpack::csr_support_view support() const {
        return {static_cast<u32>(row_offsets.size() - 1u), 6u,
            static_cast<u32>(feature_ids.size()), row_offsets.data(),
            feature_ids.empty() ? nullptr : feature_ids.data()};
    }

    cellpack::record_validation_source_view validation_view() const {
        return {dataset_identity, global_row_begin, full_row_count,
            support(), 1u,
            values.empty() ? nullptr : values.data()};
    }
};

canonical_matrix make_real_matrix() {
    canonical_matrix result;
    result.row_offsets = {0u, 2u, 4u, 6u, 8u, 10u, 12u, 14u, 16u};
    result.feature_ids = {
        0u, 1u, 0u, 2u, 1u, 2u, 0u, 1u,
        3u, 4u, 3u, 5u, 4u, 5u, 3u, 4u};
    result.values.resize(result.feature_ids.size());
    for (u32 entry = 0u; entry < result.values.size(); ++entry) {
        result.values[entry] = static_cast<unsigned char>(entry + 11u);
    }
    result.dataset_identity = 0x5245414c44415441ull;
    return result;
}

canonical_matrix make_empty_matrix() {
    canonical_matrix result;
    result.row_offsets.assign(9u, 0u);
    result.dataset_identity = 0x454d505459444154ull;
    return result;
}

struct record_storage {
    std::vector<u32> ordered_row_offsets, blocks, locals, canonical_features;
    std::vector<unsigned char> ordered_values;
    std::vector<u64> keys;
    std::vector<u32> source_order;
    std::vector<u32> row_record_offsets, record_blocks, masks, value_offsets;
    std::vector<unsigned char> values;
    cellpack::cell_block_record_view records{};
};

record_storage build_records(
    const cellpack::frozen_packing_plan &plan,
    const canonical_matrix &source) {
    record_storage storage;
    const u32 row_count = source.support().row_count;
    const u32 nnz_count = source.support().nnz_count;
    storage.ordered_row_offsets.resize(static_cast<std::size_t>(row_count) + 1u);
    storage.blocks.resize(nnz_count);
    storage.locals.resize(nnz_count);
    storage.canonical_features.resize(nnz_count);
    storage.ordered_values.resize(nnz_count);
    storage.keys.resize(nnz_count);
    storage.source_order.resize(nnz_count);

    cellpack::plan_application_context application_context;
    application_context.full_row_count = plan.row_count();
    application_context.feature_count = plan.feature_count();
    application_context.feature_axis_fingerprint =
        plan.identity().feature_axis_fingerprint;
    application_context.feature_axis_fingerprint_version =
        plan.identity().feature_axis_fingerprint_version;
    application_context.row_domain_identity = plan.identity().row_domain_identity;
    cellpack::plan_application_source_view application_source;
    application_source.row_count = row_count;
    application_source.global_row_begin = source.global_row_begin;
    application_source.feature_count = source.support().feature_count;
    application_source.nnz_count = nnz_count;
    application_source.value_size_bytes = 1u;
    application_source.row_offsets = source.row_offsets.data();
    application_source.canonical_feature_ids = source.feature_ids.empty()
        ? nullptr : source.feature_ids.data();
    application_source.values = source.values.empty() ? nullptr : source.values.data();
    cellpack::plan_application_buffers application_buffers;
    application_buffers.row_offset_capacity = storage.ordered_row_offsets.size();
    application_buffers.entry_capacity = nnz_count;
    application_buffers.value_capacity_bytes = nnz_count;
    application_buffers.row_offsets = storage.ordered_row_offsets.data();
    application_buffers.block_ids = storage.blocks.empty() ? nullptr : storage.blocks.data();
    application_buffers.local_feature_ids = storage.locals.empty() ? nullptr : storage.locals.data();
    application_buffers.canonical_feature_ids = storage.canonical_features.empty()
        ? nullptr : storage.canonical_features.data();
    application_buffers.values = storage.ordered_values.empty()
        ? nullptr : storage.ordered_values.data();
    cellpack::plan_application_host_workspace_view workspace;
    workspace.entry_capacity = nnz_count;
    workspace.keys = storage.keys.empty() ? nullptr : storage.keys.data();
    workspace.source_order = storage.source_order.empty()
        ? nullptr : storage.source_order.data();
    cellpack::ordered_plan_partition_view ordered;
    require_status(cellpack::apply_frozen_plan_host(plan, application_context,
        application_source, workspace, application_buffers, &ordered),
        "apply frozen plan");

    cellpack::cell_block_record_requirements requirements;
    require_status(cellpack::query_cell_block_record_requirements_host(
        plan, ordered, &requirements), "query record requirements");
    storage.row_record_offsets.resize(requirements.row_record_offset_count);
    storage.record_blocks.resize(requirements.record_count);
    storage.masks.resize(requirements.record_count);
    storage.value_offsets.resize(requirements.record_value_offset_count);
    storage.values.resize(requirements.value_bytes);
    cellpack::cell_block_record_buffers buffers;
    buffers.row_record_offset_capacity = storage.row_record_offsets.size();
    buffers.record_capacity = storage.record_blocks.size();
    buffers.record_value_offset_capacity = storage.value_offsets.size();
    buffers.value_capacity_bytes = storage.values.size();
    buffers.row_record_offsets = storage.row_record_offsets.data();
    buffers.record_block_ids = storage.record_blocks.empty()
        ? nullptr : storage.record_blocks.data();
    buffers.record_gene_masks = storage.masks.empty() ? nullptr : storage.masks.data();
    buffers.record_value_offsets = storage.value_offsets.data();
    buffers.values = storage.values.empty() ? nullptr : storage.values.data();
    require_status(cellpack::build_cell_block_records_host(
        plan, ordered, buffers, &storage.records), "build cell-block records");
    return storage;
}

canonical_matrix slice_matrix(
    const canonical_matrix &source,
    u32 row_begin,
    u32 row_count) {
    canonical_matrix result;
    result.dataset_identity = source.dataset_identity;
    result.global_row_begin = row_begin;
    result.full_row_count = source.full_row_count;
    result.row_offsets.resize(static_cast<std::size_t>(row_count) + 1u);
    const u32 entry_begin = source.row_offsets[row_begin];
    const u32 entry_end = source.row_offsets[row_begin + row_count];
    for (u32 row = 0u; row <= row_count; ++row) {
        result.row_offsets[row] = source.row_offsets[row_begin + row] - entry_begin;
    }
    result.feature_ids.assign(source.feature_ids.begin() + entry_begin,
        source.feature_ids.begin() + entry_end);
    result.values.assign(source.values.begin() + entry_begin,
        source.values.begin() + entry_end);
    return result;
}

struct split_fixture {
    std::vector<u64> rows{100u, 101u, 102u, 103u, 104u, 105u, 106u, 107u};
    std::vector<u64> groups{10u, 10u, 20u, 20u, 30u, 30u, 40u, 40u};
    std::vector<cellpack::validation_partition> partitions;
    cellpack::validation_split_provenance provenance{};

    split_fixture() : partitions(rows.size()) {
        const cellpack::validation_identity_view identities{
            static_cast<u32>(rows.size()), rows.data(), groups.data()};
        require_status(cellpack::build_validation_split(identities,
            {0x53504c4954ull, 2u}, {partitions.size(), partitions.data()},
            &provenance), "build group-aware split");
    }

    cellpack::record_validation_context context() const {
        cellpack::record_validation_context result;
        result.feature_axis_identity = 0x4645415455524553ull;
        result.feature_axis_identity_version = 1u;
        result.row_domain_identity = 0x524f57444f4d4149ull;
        result.plan_training_split_identity = provenance.assignment_identity;
        result.identities = {static_cast<u32>(rows.size()), rows.data(), groups.data()};
        result.row_partitions = partitions.data();
        result.split_provenance = provenance;
        return result;
    }
};

void test_group_aware_exact_record_metrics() {
    const cellpack::frozen_packing_plan plan = make_plan();
    const canonical_matrix source = make_real_matrix();
    const record_storage storage = build_records(plan, source);
    const split_fixture split;
    cellpack::held_out_record_validation first, second;
    require_status(cellpack::evaluate_held_out_cell_block_records(plan,
        split.context(), source.validation_view(), storage.records, &first),
        "evaluate group-aware held-out records");
    require_status(cellpack::evaluate_held_out_cell_block_records(plan,
        split.context(), source.validation_view(), storage.records, &second),
        "repeat group-aware held-out records");
    require(first.schema_version == cellpack::record_statistical_validation_schema_version
            && first.frozen_plan_identity != 0u
            && first.frozen_plan_identity == second.frozen_plan_identity,
        "frozen plan identity is missing or unstable");
    require(first.held_out_row_identity == second.held_out_row_identity,
        "held-out row identity is not deterministic");
    require(first.plan_training_split_identity == split.provenance.assignment_identity,
        "frozen plan was not bound to the learning split identity");
    require(first.claims_group_generalization
            && first.unit_kind == cellpack::validation_unit_kind::caller_group_identity,
        "group-aware evaluation lost generalization scope");
    require(first.metrics.row_count == 4u && first.metrics.nnz_count == 8u,
        "held-out metric denominators are wrong");
    require(first.metrics.active_block_references == 4u,
        "real clustered records should have one active block per held-out row");
    require(first.metrics.metadata_bytes == 104u
            && first.metrics.encoded_bytes == 112u
            && first.metrics.baseline_bytes == 92u,
        "held-out projection byte accounting changed");
    require(first.metrics.correctness_items == 12u
            && first.metrics.correctness_mismatches == 0u,
        "exact reconstruction metric is wrong");
    require((first.metrics.available & cellpack::packing_validation_metric_storage) != 0u
            && (first.metrics.available & cellpack::packing_validation_metric_records) != 0u
            && (first.metrics.available & cellpack::packing_validation_metric_tiles) == 0u
            && (first.metrics.available & cellpack::packing_validation_metric_runtime) == 0u,
        "record phase fabricated unavailable tile/runtime metrics");
    cellpack::packing_validation_metric_rates rates;
    require_status(cellpack::derive_packing_validation_metric_rates(
        first.metrics, &rates), "derive held-out record rates");
    require_close(rates.encoded_bytes_per_nnz, 14.0,
        "held-out encoded bytes/NNZ changed");
    require_close(rates.metadata_bytes_per_nnz, 13.0,
        "held-out metadata bytes/NNZ changed");
    require_close(rates.active_blocks_per_row, 1.0,
        "held-out blocks/cell changed");
    require(rates.exact_correctness, "held-out exact reconstruction was not reported");
}

void test_cell_level_scope_and_zero_nnz_denominators() {
    const cellpack::frozen_packing_plan plan = make_plan();
    const canonical_matrix empty = make_empty_matrix();
    const record_storage records = build_records(plan, empty);
    const u64 rows[] = {200u, 201u, 202u, 203u, 204u, 205u, 206u, 207u};
    const cellpack::validation_identity_view identities{8u, rows, nullptr};
    std::vector<cellpack::validation_partition> partitions(8u);
    cellpack::validation_split_provenance provenance;
    require_status(cellpack::build_validation_split(identities, {77u, 3u},
        {partitions.size(), partitions.data()}, &provenance),
        "build cell-level split");
    cellpack::record_validation_context context;
    context.feature_axis_identity = plan.identity().feature_axis_fingerprint;
    context.feature_axis_identity_version =
        plan.identity().feature_axis_fingerprint_version;
    context.row_domain_identity = plan.identity().row_domain_identity;
    context.plan_training_split_identity = provenance.assignment_identity;
    context.identities = identities;
    context.row_partitions = partitions.data();
    context.split_provenance = provenance;
    cellpack::held_out_record_validation result;
    require_status(cellpack::evaluate_held_out_cell_block_records(plan, context,
        empty.validation_view(), records.records, &result),
        "evaluate empty held-out support");
    require(!result.claims_group_generalization
            && result.unit_kind == cellpack::validation_unit_kind::row_identity,
        "cell-level split overclaimed donor/sample/study generalization");
    require(result.metrics.row_count == 3u && result.metrics.nnz_count == 0u
            && result.metrics.active_block_references == 0u,
        "empty held-out denominators are wrong");
    require((result.metrics.available & cellpack::packing_validation_metric_storage) == 0u
            && result.metrics.encoded_bytes != 0u
            && result.metrics.metadata_bytes == result.metrics.encoded_bytes,
        "zero-NNZ projection did not preserve raw bytes while withholding a rate");
    cellpack::packing_validation_metric_rates rates;
    require_status(cellpack::derive_packing_validation_metric_rates(
        result.metrics, &rates), "derive zero-NNZ record rates");
    require_close(rates.encoded_bytes_per_nnz, 0.0,
        "zero denominator produced a storage rate");
    require_close(rates.active_blocks_per_row, 0.0,
        "empty rows produced active blocks");
    require(rates.exact_correctness, "empty held-out rows were not exact");
}

void test_nonzero_global_partition_identity() {
    const cellpack::frozen_packing_plan plan = make_plan();
    const canonical_matrix full = make_real_matrix();
    const split_fixture split;
    u32 held_out_row = 0u;
    while (split.partitions[held_out_row]
        != cellpack::validation_partition::held_out) ++held_out_row;
    const u32 group_begin = held_out_row & ~1u;
    const canonical_matrix partition = slice_matrix(full, group_begin, 2u);
    const record_storage records = build_records(plan, partition);
    cellpack::held_out_record_validation result;
    require_status(cellpack::evaluate_held_out_cell_block_records(plan,
        split.context(), partition.validation_view(), records.records, &result),
        "evaluate nonzero global-row partition");
    require(records.records.global_row_begin == group_begin
            && result.metrics.row_count == 2u
            && result.metrics.nnz_count == 4u,
        "nonzero global-row partition lost its canonical row identity");
}

canonical_matrix build_null_matrix(
    const canonical_matrix &real,
    cellpack::degree_preserving_null_provenance *provenance) {
    canonical_matrix result;
    result.row_offsets.resize(real.row_offsets.size());
    result.feature_ids.resize(real.feature_ids.size());
    cellpack::csr_support_view output;
    require_status(cellpack::build_degree_preserving_null_reference(real.support(),
        {0x4e554c4c53454544ull, real.dataset_identity, 32u, 8192u},
        {result.row_offsets.size(), result.feature_ids.size(),
            result.row_offsets.data(), result.feature_ids.data()},
        &output, provenance), "build degree-preserving null");
    require(provenance->target_reached, "null fixture did not reach its swap target");
    result.values.assign(result.feature_ids.size(), 1u);
    result.dataset_identity = provenance->output_identity;
    return result;
}

void test_real_null_comparison() {
    const cellpack::frozen_packing_plan plan = make_plan();
    const canonical_matrix real = make_real_matrix();
    cellpack::degree_preserving_null_provenance provenance;
    const canonical_matrix null_matrix = build_null_matrix(real, &provenance);
    require(null_matrix.feature_ids != real.feature_ids,
        "degree-preserving null did not alter the real support");
    const record_storage real_records = build_records(plan, real);
    const record_storage null_records = build_records(plan, null_matrix);
    const split_fixture split;
    cellpack::held_out_record_null_comparison comparison;
    require_status(cellpack::compare_held_out_cell_block_records_to_degree_null(
        plan, split.context(), real.validation_view(), real_records.records,
        null_matrix.validation_view(), null_records.records, provenance,
        &comparison), "compare real and degree-preserving null records");
    require(comparison.exact_degree_conservation,
        "real/null comparison lost exact degree conservation");
    require(comparison.real.metrics.dataset_identity == real.dataset_identity
            && comparison.degree_preserving_null.metrics.dataset_identity
                == provenance.output_identity,
        "real/null comparison lost dataset identities");
    require(comparison.real.held_out_row_identity
            == comparison.degree_preserving_null.held_out_row_identity,
        "real/null comparison changed the held-out rows");
    require(comparison.real.metrics.nnz_count
            == comparison.degree_preserving_null.metrics.nnz_count,
        "degree-preserving null changed held-out NNZ");
    require(comparison.encoded_bytes_absolute_difference
            == (comparison.real.metrics.encoded_bytes
                    <= comparison.degree_preserving_null.metrics.encoded_bytes
                ? comparison.degree_preserving_null.metrics.encoded_bytes
                    - comparison.real.metrics.encoded_bytes
                : comparison.real.metrics.encoded_bytes
                    - comparison.degree_preserving_null.metrics.encoded_bytes),
        "real/null raw byte difference is inconsistent");
    require(comparison.active_block_references_absolute_difference != 0u
            && comparison.real_active_blocks_no_greater,
        "clustered real support did not separate from the degree-preserving null");
}

void test_tamper_and_leakage_rejection() {
    const cellpack::frozen_packing_plan plan = make_plan();
    const canonical_matrix source = make_real_matrix();
    const record_storage storage = build_records(plan, source);
    const split_fixture split;
    cellpack::held_out_record_validation ignored;

    cellpack::record_validation_context bad_context = split.context();
    bad_context.feature_axis_identity ^= 1u;
    require_code(cellpack::evaluate_held_out_cell_block_records(plan, bad_context,
        source.validation_view(), storage.records, &ignored),
        cellpack::validation_code::invalid_plan_geometry,
        "tampered feature identity was accepted");

    bad_context = split.context();
    bad_context.plan_training_split_identity ^= 1u;
    require_code(cellpack::evaluate_held_out_cell_block_records(plan, bad_context,
        source.validation_view(), storage.records, &ignored),
        cellpack::validation_code::invalid_plan_geometry,
        "mismatched plan-training split identity was accepted");

    std::vector<cellpack::validation_partition> leaked = split.partitions;
    leaked[1] = leaked[0] == cellpack::validation_partition::training
        ? cellpack::validation_partition::held_out
        : cellpack::validation_partition::training;
    bad_context = split.context();
    bad_context.row_partitions = leaked.data();
    require_code(cellpack::evaluate_held_out_cell_block_records(plan, bad_context,
        source.validation_view(), storage.records, &ignored),
        cellpack::validation_code::invalid_permutation,
        "one donor/sample/study group crossed the split");

    cellpack::record_validation_source_view shifted_source = source.validation_view();
    shifted_source.global_row_begin = 1u;
    require_code(cellpack::evaluate_held_out_cell_block_records(plan,
        split.context(), shifted_source, storage.records, &ignored),
        cellpack::validation_code::invalid_matrix_view,
        "tampered canonical source row identity was accepted");

    canonical_matrix changed_values = source;
    u32 first_held_out_row = 0u;
    while (split.partitions[first_held_out_row]
        != cellpack::validation_partition::held_out) ++first_held_out_row;
    changed_values.values[changed_values.row_offsets[first_held_out_row]] ^= 0xffu;
    require_code(cellpack::evaluate_held_out_cell_block_records(plan,
        split.context(), changed_values.validation_view(), storage.records, &ignored),
        cellpack::validation_code::invalid_matrix_view,
        "tampered held-out value bytes were accepted");

    cellpack::degree_preserving_null_provenance provenance;
    const canonical_matrix null_matrix = build_null_matrix(source, &provenance);
    const record_storage null_records = build_records(plan, null_matrix);
    ++provenance.output_identity;
    cellpack::held_out_record_null_comparison comparison;
    require(!cellpack::compare_held_out_cell_block_records_to_degree_null(
        plan, split.context(), source.validation_view(), storage.records,
        null_matrix.validation_view(), null_records.records, provenance,
        &comparison), "tampered null provenance was accepted");
}

} // namespace

int main() {
    test_group_aware_exact_record_metrics();
    test_cell_level_scope_and_zero_nnz_denominators();
    test_nonzero_global_partition_identity();
    test_real_null_comparison();
    test_tamper_and_leakage_rejection();
    return 0;
}
