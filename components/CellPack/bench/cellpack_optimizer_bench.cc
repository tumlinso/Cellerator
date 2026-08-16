#include <CellPack/optimizer.hh>

#include "benchmark_mutex.hh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct options {
    cellpack::u32 rows = 20000u;
    cellpack::u32 features = 5000u;
    cellpack::u32 nnz_per_row = 16u;
    cellpack::u32 sampled_rows = 4096u;
    cellpack::u32 maximum_block_width = 8u;
    cellpack::u32 row_group_width = 128u;
    cellpack::u32 candidate_degree = 2u;
    cellpack::u32 oracle_batch_size = 8u;
};

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

cellpack::u32 parse_u32(const char *text, const char *label) {
    if (text == nullptr || *text == '\0') throw std::invalid_argument(std::string("missing value for ") + label);
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == nullptr || *end != '\0' || value > 0xfffffffful) {
        throw std::invalid_argument(std::string("invalid uint32 value for ") + label);
    }
    return static_cast<cellpack::u32>(value);
}

options parse_options(int argc, char **argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        auto next = [&](const char *label) {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + label);
            return argv[++i];
        };
        if (argument == "--rows") result.rows = parse_u32(next("--rows"), "--rows");
        else if (argument == "--features") result.features = parse_u32(next("--features"), "--features");
        else if (argument == "--nnz-row") result.nnz_per_row = parse_u32(next("--nnz-row"), "--nnz-row");
        else if (argument == "--sampled-rows") result.sampled_rows = parse_u32(next("--sampled-rows"), "--sampled-rows");
        else if (argument == "--max-block-width") result.maximum_block_width = parse_u32(next("--max-block-width"), "--max-block-width");
        else if (argument == "--row-group-width") result.row_group_width = parse_u32(next("--row-group-width"), "--row-group-width");
        else if (argument == "--candidate-degree") result.candidate_degree = parse_u32(next("--candidate-degree"), "--candidate-degree");
        else if (argument == "--oracle-batch") result.oracle_batch_size = parse_u32(next("--oracle-batch"), "--oracle-batch");
        else if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: cellPackOptimizerBench [--rows N] [--features N] [--nnz-row N] "
                "[--sampled-rows N] [--max-block-width N] [--row-group-width N] "
                "[--candidate-degree N] [--oracle-batch N]\n";
            std::exit(0);
        } else throw std::invalid_argument("unknown argument: " + argument);
    }
    require(result.rows != 0u && result.features != 0u, "rows/features must be nonzero");
    require(result.nnz_per_row != 0u && result.nnz_per_row <= result.features, "nnz-row is invalid");
    require(result.sampled_rows != 0u && result.sampled_rows <= result.rows, "sampled rows are invalid");
    require(result.maximum_block_width != 0u && result.row_group_width != 0u, "configured widths must be nonzero");
    require(result.candidate_degree != 0u && result.oracle_batch_size != 0u, "candidate degree/batch must be nonzero");
    require(static_cast<cellpack::u64>(result.rows) * result.nnz_per_row <= 0xffffffffull,
        "benchmark NNZ exceeds uint32 evaluator limit");
    return result;
}

struct fixture {
    std::vector<cellpack::u32> row_offsets;
    std::vector<cellpack::u32> feature_ids;
    std::vector<cellpack::u32> support_words;
    std::vector<cellpack::u32> detection_counts;
    std::vector<cellpack::u64> sampled_rows;
    std::vector<cellpack::candidate_relation> candidates;
};

fixture make_fixture(const options &settings) {
    fixture result;
    const cellpack::u32 words_per_feature = 1u + ((settings.sampled_rows - 1u) / 32u);
    result.support_words.assign(static_cast<std::size_t>(settings.features) * words_per_feature, 0u);
    result.detection_counts.assign(settings.features, 0u);
    result.sampled_rows.resize(settings.sampled_rows);
    std::iota(result.sampled_rows.begin(), result.sampled_rows.end(), 0u);
    result.row_offsets.reserve(static_cast<std::size_t>(settings.rows) + 1u);
    result.feature_ids.reserve(static_cast<std::size_t>(settings.rows) * settings.nnz_per_row);
    result.row_offsets.push_back(0u);
    std::vector<cellpack::u32> row_features(settings.nnz_per_row);
    const cellpack::u32 group_width = std::max<cellpack::u32>(settings.maximum_block_width, settings.nnz_per_row);
    const cellpack::u32 group_count = std::max<cellpack::u32>(1u, settings.features / group_width);
    for (cellpack::u32 row = 0u; row < settings.rows; ++row) {
        const cellpack::u32 group = static_cast<cellpack::u32>((static_cast<cellpack::u64>(row) * 131u) % group_count);
        const cellpack::u32 begin = group * group_width;
        for (cellpack::u32 entry = 0u; entry < settings.nnz_per_row; ++entry) {
            row_features[entry] = (begin + entry) % settings.features;
        }
        std::sort(row_features.begin(), row_features.end());
        row_features.erase(std::unique(row_features.begin(), row_features.end()), row_features.end());
        for (cellpack::u32 feature : row_features) {
            result.feature_ids.push_back(feature);
            if (row < settings.sampled_rows) {
                result.support_words[static_cast<std::size_t>(feature) * words_per_feature + row / 32u]
                    |= cellpack::u32{1u} << (row % 32u);
                ++result.detection_counts[feature];
            }
        }
        result.row_offsets.push_back(static_cast<cellpack::u32>(result.feature_ids.size()));
        row_features.resize(settings.nnz_per_row);
    }
    for (cellpack::u32 feature = 0u; feature < settings.features; ++feature) {
        for (cellpack::u32 distance = 1u; distance <= settings.candidate_degree; ++distance) {
            if (feature + distance >= settings.features) break;
            cellpack::u64 intersection = 0u;
            for (cellpack::u32 word = 0u; word < words_per_feature; ++word) {
                const cellpack::u32 lhs = result.support_words[static_cast<std::size_t>(feature) * words_per_feature + word];
                const cellpack::u32 rhs = result.support_words[static_cast<std::size_t>(feature + distance) * words_per_feature + word];
                intersection += static_cast<cellpack::u64>(__builtin_popcount(lhs & rhs));
            }
            cellpack::candidate_relation candidate;
            candidate.feature_a = feature;
            candidate.feature_b = feature + distance;
            candidate.score_numerator = static_cast<std::int64_t>(intersection);
            candidate.score_denominator = 1u;
            candidate.score_kind = cellpack::candidate_score_kind::support_intersection;
            candidate.evidence_flags = cellpack::candidate_evidence_exact
                | cellpack::candidate_evidence_support_counts
                | cellpack::candidate_evidence_intersection;
            candidate.support_a = result.detection_counts[feature];
            candidate.support_b = result.detection_counts[feature + distance];
            candidate.support_intersection = intersection;
            result.candidates.push_back(candidate);
        }
    }
    return result;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const options settings = parse_options(argc, argv);
        cellerator::bench::benchmark_mutex_guard mutex("cellPackOptimizerBench");
        fixture data = make_fixture(settings);
        const cellpack::csr_support_view csr{settings.rows, settings.features,
            static_cast<cellpack::u32>(data.feature_ids.size()), data.row_offsets.data(), data.feature_ids.data()};
        cellpack::prepared_csr_support prepared;
        cellpack::validation_result status = cellpack::prepare_csr_support(csr, &prepared);
        require(static_cast<bool>(status), status.message);
        const cellpack::u32 words_per_feature = 1u + ((settings.sampled_rows - 1u) / 32u);
        const cellpack::sampled_feature_support_view support{settings.sampled_rows, settings.features,
            words_per_feature, data.support_words.data(), data.detection_counts.data(), data.sampled_rows.data()};
        cellpack::packing_optimizer_workspace_requirements requirements;
        status = cellpack::query_packing_optimizer_workspace_requirements(prepared, settings.row_group_width, &requirements);
        require(static_cast<bool>(status), status.message);
        std::vector<cellpack::packing_evaluation_entry> entries(requirements.evaluator.workspace_entry_capacity);
        std::vector<cellpack::occupied_tile_occupancy> tiles(requirements.evaluator.occupied_tile_capacity);
        std::vector<cellpack::u32> row_active(requirements.evaluator.execution_row_capacity);
        std::vector<cellpack::row_group_occupancy> row_groups(requirements.evaluator.row_group_capacity);
        const cellpack::packing_optimizer_workspace_view workspace{
            {entries.data(), static_cast<cellpack::u32>(entries.size())},
            {tiles.data(), static_cast<cellpack::u32>(tiles.size()),
             row_active.data(), static_cast<cellpack::u32>(row_active.size()),
             row_groups.data(), static_cast<cellpack::u32>(row_groups.size())}
        };
        cellpack::packing_optimizer_config config;
        config.maximum_feature_block_width = settings.maximum_block_width;
        config.row_group_width = settings.row_group_width;
        config.candidate_fanout = std::max<cellpack::u32>(settings.candidate_degree * 2u, 2u);
        config.proposal_shortlist = std::min<cellpack::u32>(settings.features * 2u, 16384u);
        config.initial_oracle_batch_size = settings.oracle_batch_size;
        config.maximum_coarsening_passes = 32u;
        config.maximum_refinement_passes = 4u;
        config.maximum_oracle_evaluations = 128u;
        config.objective_kind = cellpack::packing_exact_objective_kind::row_active_block_references;
        config.cost_policy_identity = 0x4350425030340001ull;
        config.plan_identity.feature_axis_fingerprint = 0x4654585f42454e43ull;
        config.plan_identity.feature_axis_fingerprint_version = 1u;
        config.plan_identity.row_domain_kind = cellpack::packing_row_domain_kind::sampled_rows_identity;
        config.plan_identity.row_domain_identity = 0x524f575f42454e43ull;
        config.plan_identity.evaluation_source_identity = 0x4556414c5f42454eull;
        config.plan_identity.sampling_provenance_identity = 0x53414d505f42454eull;
        cellpack::packing_optimizer_result result;
        status = cellpack::optimize_packing_plan(prepared, support,
            {data.candidates.data(), data.candidates.size()}, config, workspace, &result);
        require(static_cast<bool>(status), status.message);

        std::map<cellpack::u32, cellpack::u32> width_histogram;
        for (cellpack::u32 block = 0u; block < result.plan.feature_block_count(); ++block) {
            const cellpack::u32 width = result.plan.feature_block_offsets()[block + 1u]
                - result.plan.feature_block_offsets()[block];
            ++width_histogram[width];
        }
        const cellpack::packing_optimizer_diagnostics &d = result.diagnostics;
        const double oracle_fraction = d.total_ms == 0.0 ? 0.0 : d.oracle_ms / d.total_ms;
        std::cout << "optimizer: cp_bp_04_proxy_plus_cpu_oracle\n";
        std::cout << "features: " << settings.features << "\n";
        std::cout << "sampled_rows: " << settings.sampled_rows << "\n";
        std::cout << "support_words_per_feature: " << words_per_feature << "\n";
        std::cout << "support_bytes: " << data.support_words.size() * sizeof(cellpack::u32) << "\n";
        std::cout << "evaluator_rows: " << settings.rows << "\n";
        std::cout << "evaluator_nnz: " << csr.nnz_count << "\n";
        std::cout << "candidate_edges_before: " << data.candidates.size() << "\n";
        std::cout << "candidate_edges_after: " << d.candidate_normalization.output_relations << "\n";
        std::cout << "maximum_block_width: " << settings.maximum_block_width << "\n";
        std::cout << "row_group_width: " << settings.row_group_width << "\n";
        std::cout << "initial_blocks: " << d.initial_block_count << "\n";
        std::cout << "final_blocks: " << d.final_block_count << "\n";
        std::cout << "block_width_histogram:";
        for (const auto &entry : width_histogram) std::cout << " " << entry.first << ":" << entry.second;
        std::cout << "\n";
        std::cout << "coarsening_passes: " << d.coarsening_passes << "\n";
        std::cout << "refinement_passes: " << d.refinement_passes << "\n";
        std::cout << "merge_proposals_considered: " << d.merge_proposals_considered << "\n";
        std::cout << "merge_proposals_shortlisted: " << d.merge_proposals_shortlisted << "\n";
        std::cout << "merge_proxy_positive: " << d.merge_proxy_positive << "\n";
        std::cout << "merge_oracle_accepted: " << d.merge_oracle_accepted << "\n";
        std::cout << "merge_oracle_rejected: " << d.merge_oracle_rejected << "\n";
        std::cout << "move_proposals_considered: " << d.move_proposals_considered << "\n";
        std::cout << "move_oracle_accepted: " << d.move_oracle_accepted << "\n";
        std::cout << "move_oracle_rejected: " << d.move_oracle_rejected << "\n";
        std::cout << "swap_proposals_considered: " << d.swap_proposals_considered << "\n";
        std::cout << "swap_oracle_accepted: " << d.swap_oracle_accepted << "\n";
        std::cout << "swap_oracle_rejected: " << d.swap_oracle_rejected << "\n";
        std::cout << "full_oracle_evaluations: " << d.oracle_evaluations << "\n";
        std::cout << "initial_exact_objective: " << d.baseline.objective << "\n";
        std::cout << "final_exact_objective: " << d.final.objective << "\n";
        std::cout << "candidate_processing_ms: " << d.candidate_processing_ms << "\n";
        std::cout << "proxy_ms: " << d.proxy_ms << "\n";
        std::cout << "oracle_ms: " << d.oracle_ms << "\n";
        std::cout << "freeze_ms: " << d.freeze_ms << "\n";
        std::cout << "total_ms: " << d.total_ms << "\n";
        std::cout << "oracle_fraction: " << oracle_fraction << "\n";
        std::cout << "peak_additional_optimizer_bytes: " << d.peak_additional_optimizer_bytes << "\n";
        std::cout << "evaluator_workspace_bytes: " << requirements.evaluator.temporary_workspace_bytes << "\n";
        std::cout << "gpu_evaluator_route: deferred_cub_sm70\n";
    } catch (const std::exception &error) {
        std::cerr << "cellPackOptimizerBench: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
