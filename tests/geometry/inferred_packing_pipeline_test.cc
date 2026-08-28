#include "Cellerator/geometry/apply_plan.hh"
#include "Cellerator/geometry/merge_cost.hh"
#include "Cellerator/geometry/optimizer.hh"

#include <Cellerator/geometry/gene_candidate_discovery.hh>
#include <Cellerator/geometry/gene_support_bitset.hh>
#include <Cellerator/compute/sampling_materialization.hh>

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace candidates = ::cellerator::compute::gene_candidates;
namespace gene_support = ::cellerator::compute::gene_support;
namespace matrix = ::cellerator::matrix;
namespace sampling = ::cellerator::compute::sampling;
namespace types = ::cellerator::types;

using cellpack::u32;
using cellpack::u64;

constexpr u64 feature_axis_fingerprint = 0x4350425030303035ull;
constexpr u64 full_row_domain_identity = 0x4350425046554c4cull;
constexpr u64 evaluation_source_identity = 0x435042504556414cull;
constexpr u64 optimizer_cost_policy_identity = 0x435042504f524143ull;

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "cellPackInferredPackingPipelineTest: " << message << '\n';
    std::exit(1);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

void require_status(const cellpack::validation_result &status, const char *context) {
    if (!status) fail(std::string(context) + ": " + status.message);
}

bool fill_full_source(matrix::compressed *source) {
    const std::vector<types::ptr_t> row_offsets{0u, 3u, 5u, 7u, 10u, 12u, 14u, 17u, 20u};
    const std::vector<types::idx_t> features{
        0u, 1u, 4u,
        0u, 1u,
        2u, 3u,
        2u, 3u, 5u,
        0u, 1u,
        2u, 3u,
        0u, 1u, 4u,
        2u, 3u, 5u
    };
    matrix::init(source, 8u, 6u, static_cast<types::nnz_t>(features.size()), matrix::compressed_by_row);
    if (!matrix::allocate(source)) return false;
    std::copy(row_offsets.begin(), row_offsets.end(), source->majorPtr);
    for (std::size_t entry = 0u; entry < features.size(); ++entry) {
        source->minorIdx[entry] = features[entry];
        source->val[entry] = __float2half(static_cast<float>(entry + 1u));
    }
    return true;
}

bool contains_pair(const candidates::gene_candidate_pair_view &view, u32 lhs, u32 rhs) {
    if (lhs > rhs) std::swap(lhs, rhs);
    for (u64 index = 0u; index < view.count; ++index) {
        if (view.pairs[index].gene_a == lhs && view.pairs[index].gene_b == rhs) return true;
    }
    return false;
}

struct optimizer_workspace {
    std::vector<cellpack::packing_evaluation_entry> entries;
    std::vector<cellpack::occupied_tile_occupancy> occupied_tiles;
    std::vector<u32> active_blocks_per_row;
    std::vector<cellpack::row_group_occupancy> row_groups;

    optimizer_workspace(const cellpack::prepared_csr_support &source, u32 row_group_width) {
        cellpack::packing_optimizer_workspace_requirements requirements;
        require_status(cellpack::query_packing_optimizer_workspace_requirements(
            source, row_group_width, &requirements), "query optimizer workspace");
        entries.resize(requirements.evaluator.workspace_entry_capacity);
        occupied_tiles.resize(requirements.evaluator.occupied_tile_capacity);
        active_blocks_per_row.resize(requirements.evaluator.execution_row_capacity);
        row_groups.resize(requirements.evaluator.row_group_capacity);
    }

    cellpack::packing_optimizer_workspace_view view() {
        cellpack::packing_optimizer_workspace_view result;
        result.evaluator_workspace.entries = entries.empty() ? nullptr : entries.data();
        result.evaluator_workspace.entry_capacity = static_cast<u32>(entries.size());
        result.occupancy_buffers.occupied_tiles = occupied_tiles.empty() ? nullptr : occupied_tiles.data();
        result.occupancy_buffers.occupied_tile_capacity = static_cast<u32>(occupied_tiles.size());
        result.occupancy_buffers.active_feature_blocks_per_execution_row =
            active_blocks_per_row.empty() ? nullptr : active_blocks_per_row.data();
        result.occupancy_buffers.execution_row_capacity = static_cast<u32>(active_blocks_per_row.size());
        result.occupancy_buffers.row_groups = row_groups.empty() ? nullptr : row_groups.data();
        result.occupancy_buffers.row_group_capacity = static_cast<u32>(row_groups.size());
        return result;
    }
};

void require_exact_round_trip(
    const matrix::compressed &source,
    const cellpack::frozen_packing_plan &plan,
    const cellpack::ordered_plan_partition_view &ordered,
    const std::vector<types::idx_t> &ordered_features,
    const std::vector<::cellerator::real::storage_t> &ordered_values) {
    require(ordered.row_count == source.rows && ordered.nnz_count == source.nnz,
        "ordered full-partition shape mismatch");
    for (u32 row = 0u; row < source.rows; ++row) {
        const u32 begin = ordered.row_offsets[row], end = ordered.row_offsets[row + 1u];
        u64 previous_key = 0u;
        bool have_previous = false;
        for (u32 output_entry = begin; output_entry < end; ++output_entry) {
            const u32 canonical = ordered_features[output_entry];
            const u32 block = ordered.block_ids[output_entry];
            const u32 local = ordered.local_feature_ids[output_entry];
            require(block == plan.feature_to_block()[canonical]
                    && local == plan.feature_to_local()[canonical],
                "ordered packed coordinate disagrees with frozen plan");
            const u64 key = (static_cast<u64>(block) << 32u) | local;
            require(!have_previous || previous_key < key, "ordered row is not strictly packed-coordinate sorted");
            previous_key = key;
            have_previous = true;

            bool found = false;
            for (u32 source_entry = source.majorPtr[row]; source_entry < source.majorPtr[row + 1u]; ++source_entry) {
                if (source.minorIdx[source_entry] != canonical) continue;
                require(std::memcmp(&ordered_values[output_entry], &source.val[source_entry],
                            sizeof(::cellerator::real::storage_t)) == 0,
                    "ordered value bytes changed during application");
                found = true;
                break;
            }
            require(found, "ordered canonical feature is absent from source row");
        }
        require(end - begin == source.majorPtr[row + 1u] - source.majorPtr[row],
            "ordered row changed nonzero count");
    }
}

void run_pipeline() {
    matrix::compressed source;
    matrix::init(&source);
    require(fill_full_source(&source), "failed to allocate full CSR fixture");

    sampling::sample_spec sample_spec;
    sample_spec.mode = sampling::selection_mode::exact_lowest_hash;
    sample_spec.seed = 0x123456789abcdef0ull;
    sample_spec.split_name = "cp-bp-00-05-integration";
    sample_spec.requested_row_count = 6u;
    sampling::sample_plan sample_plan;
    sampling::cell_identity_view cell_identities;
    std::string error;
    require(sampling::build_sample_plan(source.rows, sample_spec, cell_identities, &sample_plan, &error), error);

    sampling::owned_sampled_csr_structure sampled;
    require(sampling::materialize_sampled_csr_structure(&source, sample_plan, &sampled, &error), error);
    require(sampled.view().sampled_row_count == 6u
            && sampled.view().provenance->total_rows == source.rows,
        "sample materialization lost full-domain provenance");

    gene_support::owned_gene_support_bitsets support;
    require(gene_support::build_gene_support_bitsets_cpu(sampled.view(), &support, &error), error);
    const gene_support::gene_support_bitset_view support_view = support.view();

    candidates::candidate_discovery_config candidate_config;
    candidate_config.seed = 0x9e3779b97f4a7c15ull;
    candidate_config.sketch_count = 16u;
    candidate_config.lsh_bands = 4u;
    candidate_config.rows_per_band = 4u;
    candidate_config.maximum_bucket_size = 16u;
    candidates::owned_gene_candidates discovered;
    require(candidates::discover_gene_candidates_cpu(
        support_view, candidate_config, &discovered, &error), error);
    const candidates::gene_candidate_pair_view candidate_view = discovered.view();
    require(contains_pair(candidate_view, 0u, 1u) && contains_pair(candidate_view, 2u, 3u),
        "candidate discovery omitted identical-support feature pairs");

    cellpack::exact_merge_cost_policy merge_policy;
    merge_policy.maximum_block_width = 2u;
    cellpack::owned_exact_gene_merge_scores scored;
    require_status(cellpack::score_gene_merges_cpu(
        support_view, candidate_view, merge_policy, &scored), "score exact gene merges");
    const cellpack::exact_gene_merge_score_view scored_view = scored.view();
    require(scored_view.count == candidate_view.count && scored_view.provenance != nullptr,
        "exact scorer did not preserve candidate cardinality/provenance");
    for (u64 index = 0u; index < scored_view.count; ++index) {
        require(scored_view.relations[index].score_kind == cellpack::candidate_score_kind::exact_merge_gain
                && (scored_view.relations[index].evidence_flags & cellpack::candidate_evidence_exact) != 0u,
            "exact scorer emitted non-exact optimizer evidence");
    }

    cellpack::csr_support_view full_support;
    full_support.row_count = source.rows;
    full_support.feature_count = source.cols;
    full_support.nnz_count = source.nnz;
    full_support.row_offsets = source.majorPtr;
    full_support.feature_ids = source.minorIdx;
    cellpack::prepared_csr_support prepared;
    require_status(cellpack::prepare_csr_support(full_support, &prepared), "prepare full evaluator source");
    cellpack::sampled_feature_support_view sampled_support;
    require_status(cellpack::make_sampled_feature_support_view(
        support_view, &sampled_support), "adapt sampled feature support");
    u64 sampling_identity = 0u;
    require_status(cellpack::query_sampled_feature_support_identity(
        sampled_support, &sampling_identity), "identify sampled feature support");

    cellpack::packing_optimizer_config optimizer_config;
    optimizer_config.maximum_feature_block_width = 2u;
    optimizer_config.row_group_width = 4u;
    optimizer_config.candidate_fanout = 4u;
    optimizer_config.proposal_shortlist = 16u;
    optimizer_config.initial_oracle_batch_size = 2u;
    optimizer_config.maximum_coarsening_passes = 8u;
    optimizer_config.maximum_refinement_passes = 2u;
    optimizer_config.maximum_oracle_evaluations = 64u;
    optimizer_config.objective_kind = cellpack::packing_exact_objective_kind::row_active_block_references;
    optimizer_config.cost_policy_identity = optimizer_cost_policy_identity;
    optimizer_config.plan_identity.feature_axis_fingerprint = feature_axis_fingerprint;
    optimizer_config.plan_identity.feature_axis_fingerprint_version = 1u;
    optimizer_config.plan_identity.row_domain_kind = cellpack::packing_row_domain_kind::full_dataset_identity;
    optimizer_config.plan_identity.row_domain_identity = full_row_domain_identity;
    optimizer_config.plan_identity.evaluation_source_identity = evaluation_source_identity;
    optimizer_config.plan_identity.sampling_provenance_identity = sampling_identity;

    optimizer_workspace workspace(prepared, optimizer_config.row_group_width);
    cellpack::packing_optimizer_result optimized;
    cellpack::packing_optimizer_config wrong_sampling_identity = optimizer_config;
    wrong_sampling_identity.plan_identity.sampling_provenance_identity ^= 1u;
    require(!cellpack::optimize_packing_plan(
        prepared, sampled_support, scored_view.relation_view(),
        wrong_sampling_identity, workspace.view(), &optimized),
        "optimizer accepted a mismatched sampling provenance identity");

    cellpack::csr_support_view incomplete_support = full_support;
    incomplete_support.row_count = source.rows - 1u;
    incomplete_support.nnz_count = source.majorPtr[incomplete_support.row_count];
    cellpack::prepared_csr_support incomplete_prepared;
    require_status(cellpack::prepare_csr_support(
        incomplete_support, &incomplete_prepared), "prepare incomplete evaluator source");
    optimizer_workspace incomplete_workspace(incomplete_prepared, optimizer_config.row_group_width);
    require(!cellpack::optimize_packing_plan(
        incomplete_prepared, sampled_support, scored_view.relation_view(),
        optimizer_config, incomplete_workspace.view(), &optimized),
        "optimizer accepted incomplete evaluator rows as the full row domain");

    require_status(cellpack::optimize_packing_plan(
        prepared, sampled_support, scored_view.relation_view(),
        optimizer_config, workspace.view(), &optimized), "optimize exact scored candidates");
    require(optimized.plan.identity().row_domain_kind
            == cellpack::packing_row_domain_kind::full_dataset_identity,
        "optimizer did not freeze a full-domain plan");
    require(optimized.plan.row_count() == source.rows
            && optimized.plan.feature_block_count() < source.cols,
        "optimizer output did not preserve full rows or accept useful feature geometry");
    require(optimized.diagnostics.final.occupancy.row_active_block_references
            < optimized.diagnostics.baseline.occupancy.row_active_block_references,
        "optimizer exact oracle accepted no global improvement");

    cellpack::plan_application_context context;
    context.full_row_count = source.rows;
    context.feature_count = source.cols;
    context.feature_axis_fingerprint = feature_axis_fingerprint;
    context.feature_axis_fingerprint_version = 1u;
    context.row_domain_identity = full_row_domain_identity;
    cellpack::plan_application_source_view application_source;
    application_source.global_row_begin = 0u;
    application_source.row_count = source.rows;
    application_source.feature_count = source.cols;
    application_source.nnz_count = source.nnz;
    application_source.value_size_bytes = sizeof(::cellerator::real::storage_t);
    application_source.row_offsets = source.majorPtr;
    application_source.canonical_feature_ids = source.minorIdx;
    application_source.values = source.val;

    std::vector<u32> output_row_offsets(static_cast<std::size_t>(source.rows) + 1u);
    std::vector<u32> output_blocks(source.nnz), output_locals(source.nnz), output_features(source.nnz);
    std::vector<::cellerator::real::storage_t> output_values(source.nnz);
    std::vector<u64> keys(source.nnz);
    std::vector<u32> source_order(source.nnz);
    cellpack::plan_application_host_workspace_view application_workspace;
    application_workspace.entry_capacity = source.nnz;
    application_workspace.keys = keys.data();
    application_workspace.source_order = source_order.data();
    cellpack::plan_application_buffers output_buffers;
    output_buffers.row_offset_capacity = output_row_offsets.size();
    output_buffers.entry_capacity = source.nnz;
    output_buffers.value_capacity_bytes = output_values.size() * sizeof(output_values[0]);
    output_buffers.row_offsets = output_row_offsets.data();
    output_buffers.block_ids = output_blocks.data();
    output_buffers.local_feature_ids = output_locals.data();
    output_buffers.canonical_feature_ids = output_features.data();
    output_buffers.values = output_values.data();
    cellpack::ordered_plan_partition_view ordered;
    require_status(cellpack::apply_frozen_plan_host(
        optimized.plan, context, application_source, application_workspace,
        output_buffers, &ordered), "apply optimized full-domain plan");
    require_exact_round_trip(source, optimized.plan, ordered, output_features, output_values);

    matrix::clear(&source);
}

} // namespace

int main() {
    run_pipeline();
    return 0;
}
