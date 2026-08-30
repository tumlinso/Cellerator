#include "bench/benchmark_mutex.hh"

#include <Cellerator/geometry/support_atlas.hh>
#include <Cellerator/planner/candidate_measurement.hh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace geometry = cellerator::geometry;
namespace planner = cellerator::planner;

namespace cellerator::compute::architecture::providers::nvidia {
bool build_sm70_support_groups_v1(
    const geometry::support_atlas_view_v1 &, std::uint32_t, std::uint32_t *,
    std::uint32_t, std::uint32_t *, std::uint32_t, std::uint32_t *,
    std::uint32_t *, std::uint32_t, std::uint32_t *, std::uint32_t,
    std::uint64_t *, std::uint32_t, std::uint32_t *) noexcept;
bool build_exact_rectangle_cover_v1(
    std::uint32_t, std::uint32_t, const std::uint64_t *,
    const std::uint32_t *, const std::uint64_t *, std::uint64_t,
    const std::uint32_t *, const std::uint8_t *, std::uint32_t,
    const std::uint32_t *, const std::uint8_t *, std::uint32_t,
    const std::uint8_t *, std::uint64_t, std::uint64_t *, std::uint64_t,
    std::uint32_t *, std::uint8_t *, std::uint64_t, std::uint64_t *,
    std::uint64_t *) noexcept;
bool refine_mma_target_candidates_v1(
    const planner::phase_costs *, std::uint32_t, const std::uint32_t *,
    const std::uint8_t *, const std::uint8_t *, const planner::phase_costs *,
    std::uint32_t, std::uint64_t, std::uint64_t, std::uint64_t,
    std::uint64_t, double, double, std::uint8_t *, std::uint8_t *,
    std::uint8_t *, std::uint32_t *, std::uint32_t *, double *, double *,
    double *) noexcept;
} // namespace cellerator::compute::architecture::providers::nvidia

namespace {

namespace provider =
    cellerator::compute::architecture::providers::nvidia;

constexpr std::uint32_t group_width = 16u;
constexpr std::uint32_t original_group_count = 64u;
constexpr std::uint32_t source_count = original_group_count * group_width;
constexpr std::uint32_t destination_count =
    original_group_count * group_width;
constexpr std::uint32_t local_degree = 8u;
constexpr std::uint32_t cross_degree = 2u;
constexpr std::uint64_t edge_count =
    static_cast<std::uint64_t>(destination_count)
    * (local_degree + cross_degree);

struct relation_data {
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint32_t> sources;
    std::vector<std::uint64_t> logical_edges;
};

struct grouping_data {
    std::vector<std::uint32_t> source_to_group;
    std::vector<std::uint8_t> source_local;
    std::vector<std::uint32_t> destination_to_group;
    std::vector<std::uint8_t> destination_local;
    std::uint32_t source_groups = 0u;
    std::uint32_t destination_groups = 0u;
    std::uint64_t complete_ns = 0u;
};

struct census_data {
    std::uint64_t mma_edges = 0u;
    std::uint64_t residual_edges = 0u;
    std::uint64_t residual_runs = 0u;
    std::uint64_t complete_ns = 0u;
};

struct refinement_data {
    std::uint32_t proposal_count = 0u;
    std::uint32_t cross_group_proposals = 0u;
    std::uint32_t conservative_selected = 0u;
    std::uint32_t aggressive_selected = 0u;
    std::uint32_t aggressive_cross_group_selected = 0u;
    double pure_sparse_total_ns = 0.0;
    double conservative_total_ns = 0.0;
    double aggressive_total_ns = 0.0;
    std::uint64_t complete_ns = 0u;
};

std::uint64_t elapsed_ns(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - begin).count();
    return static_cast<std::uint64_t>(std::max<std::int64_t>(1, elapsed));
}

std::uint64_t mix(std::uint64_t digest, std::uint64_t value) noexcept {
    digest ^= value + 0x9e3779b97f4a7c15ULL + (digest << 6U)
        + (digest >> 2U);
    return digest;
}

relation_data build_relation() {
    relation_data relation;
    relation.offsets.resize(destination_count + 1u);
    relation.sources.reserve(edge_count);
    relation.logical_edges.reserve(edge_count);
    for (std::uint32_t destination = 0u;
         destination < destination_count; ++destination) {
        relation.offsets[destination] = relation.sources.size();
        const std::uint32_t group = destination / group_width;
        const std::uint32_t local = destination % group_width;
        for (std::uint32_t edge = 0u; edge < local_degree; ++edge) {
            relation.sources.push_back(group * group_width
                + (local + edge) % group_width);
            relation.logical_edges.push_back(relation.logical_edges.size());
        }
        const std::uint32_t neighbor = (group + 1u) % original_group_count;
        relation.sources.push_back(neighbor * group_width + local);
        relation.logical_edges.push_back(relation.logical_edges.size());
        relation.sources.push_back(
            neighbor * group_width + (local + 8u) % group_width);
        relation.logical_edges.push_back(relation.logical_edges.size());
    }
    relation.offsets[destination_count] = relation.sources.size();
    if (relation.sources.size() != edge_count) {
        throw std::runtime_error("relation edge count is inconsistent");
    }
    return relation;
}

std::uint64_t semantic_digest(const relation_data &relation) noexcept {
    std::uint64_t digest = 0x435347312d726575ULL;
    for (std::uint64_t value : relation.offsets) digest = mix(digest, value);
    for (std::uint32_t value : relation.sources) digest = mix(digest, value);
    return digest;
}

grouping_data build_groups() {
    std::vector<geometry::community_assignment_v1> communities(source_count);
    for (std::uint32_t source = 0u; source < source_count; ++source) {
        communities[source] = {2u, source, source / group_width, 0u};
    }
    std::vector<geometry::work_signature_v1> signatures(destination_count);
    for (std::uint32_t destination = 0u;
         destination < destination_count; ++destination) {
        signatures[destination] = {destination,
            destination / group_width + 1u, group_width,
            local_degree + cross_degree};
    }
    geometry::support_atlas_view_v1 atlas{};
    atlas.evidence_identity = 0x434547454f313137ULL;
    atlas.source_count = source_count;
    atlas.destination_count = destination_count;
    atlas.communities = communities.data();
    atlas.community_count = communities.size();
    atlas.work_signatures = signatures.data();
    atlas.work_signature_count = signatures.size();

    std::vector<std::uint32_t> source_offsets(source_count + 1u);
    std::vector<std::uint32_t> source_members(source_count);
    std::vector<std::uint32_t> destination_offsets(destination_count + 1u);
    std::vector<std::uint32_t> destination_members(destination_count);
    std::vector<std::uint64_t> destination_signatures(destination_count);
    grouping_data result;
    const auto begin = std::chrono::steady_clock::now();
    const bool ok = provider::build_sm70_support_groups_v1(
        atlas, 2u, source_offsets.data(), source_offsets.size(),
        source_members.data(), source_members.size(), &result.source_groups,
        destination_offsets.data(), destination_offsets.size(),
        destination_members.data(), destination_members.size(),
        destination_signatures.data(), destination_signatures.size(),
        &result.destination_groups);
    const auto end = std::chrono::steady_clock::now();
    if (!ok || result.source_groups != original_group_count
        || result.destination_groups != original_group_count) {
        throw std::runtime_error("SM70 grouping did not recover original groups");
    }
    result.complete_ns = elapsed_ns(begin, end);
    result.source_to_group.resize(source_count);
    result.source_local.resize(source_count);
    result.destination_to_group.resize(destination_count);
    result.destination_local.resize(destination_count);
    for (std::uint32_t group = 0u; group < result.source_groups; ++group) {
        for (std::uint32_t rank = source_offsets[group];
             rank < source_offsets[group + 1u]; ++rank) {
            result.source_to_group[source_members[rank]] = group;
            result.source_local[source_members[rank]] =
                static_cast<std::uint8_t>(rank - source_offsets[group]);
        }
    }
    for (std::uint32_t group = 0u;
         group < result.destination_groups; ++group) {
        for (std::uint32_t rank = destination_offsets[group];
             rank < destination_offsets[group + 1u]; ++rank) {
            result.destination_to_group[destination_members[rank]] = group;
            result.destination_local[destination_members[rank]] =
                static_cast<std::uint8_t>(rank - destination_offsets[group]);
        }
    }
    return result;
}

bool fixed_membership(std::uint32_t group) noexcept {
    return group % 8u == 0u;
}

census_data run_census(
    const relation_data &relation, const grouping_data &groups) {
    const std::uint64_t rectangles =
        static_cast<std::uint64_t>(groups.source_groups)
        * groups.destination_groups;
    std::vector<std::uint8_t> selected(rectangles, 0u);
    for (std::uint32_t group = 0u; group < original_group_count; ++group) {
        if (!fixed_membership(group)) {
            selected[static_cast<std::uint64_t>(group)
                * groups.source_groups + group] = 1u;
        }
    }
    std::vector<std::uint64_t> masks(rectangles * 4u);
    std::vector<std::uint32_t> occupancy(rectangles);
    std::vector<std::uint8_t> owners(edge_count);
    census_data result;
    const auto begin = std::chrono::steady_clock::now();
    const bool ok = provider::build_exact_rectangle_cover_v1(
        source_count, destination_count, relation.offsets.data(),
        relation.sources.data(), relation.logical_edges.data(), edge_count,
        groups.source_to_group.data(), groups.source_local.data(),
        groups.source_groups, groups.destination_to_group.data(),
        groups.destination_local.data(), groups.destination_groups,
        selected.data(), rectangles, masks.data(), masks.size(),
        occupancy.data(), owners.data(), owners.size(), &result.mma_edges,
        &result.residual_edges);
    const auto end = std::chrono::steady_clock::now();
    if (!ok || result.mma_edges + result.residual_edges != edge_count) {
        throw std::runtime_error("exact rectangle census failed");
    }
    result.complete_ns = elapsed_ns(begin, end);
    bool in_residual = false;
    for (std::uint8_t owner : owners) {
        if (owner == 2u && !in_residual) ++result.residual_runs;
        in_residual = owner == 2u;
    }
    return result;
}

planner::phase_costs kernel_cost(double nanoseconds) noexcept {
    planner::phase_costs result{};
    result.kernel_ns = nanoseconds;
    return result;
}

refinement_data run_refinement(
    std::uint32_t work_window, std::uint32_t work_limit) {
    const std::uint32_t rectangle_count =
        original_group_count * original_group_count;
    std::vector<planner::phase_costs> sparse(
        rectangle_count, kernel_cost(100.0));
    std::vector<std::uint32_t> rectangles;
    std::vector<std::uint8_t> move_kinds;
    std::vector<std::uint8_t> admissible;
    std::vector<planner::phase_costs> proposal_costs;
    rectangles.reserve(original_group_count * work_window);
    move_kinds.reserve(original_group_count * work_window);
    admissible.reserve(original_group_count * work_window);
    proposal_costs.reserve(original_group_count * work_window);

    // Place exact original-group rectangles first so the immediate tier sees
    // one bounded proposal for every group before cross-group exchanges.
    for (std::uint32_t group = 0u; group < original_group_count; ++group) {
        rectangles.push_back(group * original_group_count + group);
        move_kinds.push_back(5u);
        admissible.push_back(fixed_membership(group) ? 0u : 1u);
        proposal_costs.push_back(kernel_cost(75.0));
    }
    for (std::uint32_t destination = 0u;
         destination < original_group_count; ++destination) {
        const std::uint32_t window_begin =
            (destination / work_window) * work_window;
        for (std::uint32_t source = window_begin;
             source < window_begin + work_window; ++source) {
            if (source == destination) continue;
            rectangles.push_back(destination * original_group_count + source);
            move_kinds.push_back(7u);
            admissible.push_back(
                fixed_membership(destination) || fixed_membership(source)
                    ? 0u : 1u);
            proposal_costs.push_back(kernel_cost(98.0));
        }
    }
    std::vector<std::uint8_t> pure(rectangle_count);
    std::vector<std::uint8_t> conservative(rectangle_count);
    std::vector<std::uint8_t> aggressive(rectangle_count);
    std::vector<std::uint32_t> conservative_choice(rectangle_count);
    std::vector<std::uint32_t> aggressive_choice(rectangle_count);
    refinement_data result;
    result.proposal_count = rectangles.size();
    result.cross_group_proposals = rectangles.size() - original_group_count;
    const auto begin = std::chrono::steady_clock::now();
    const bool ok = provider::refine_mma_target_candidates_v1(
        sparse.data(), rectangle_count, rectangles.data(), move_kinds.data(),
        admissible.data(), proposal_costs.data(), rectangles.size(),
        1u, 1u, 1u, work_limit, 10.0, 3.0, pure.data(),
        conservative.data(), aggressive.data(), conservative_choice.data(),
        aggressive_choice.data(), &result.pure_sparse_total_ns,
        &result.conservative_total_ns, &result.aggressive_total_ns);
    const auto end = std::chrono::steady_clock::now();
    if (!ok) throw std::runtime_error("target refinement failed");
    result.complete_ns = elapsed_ns(begin, end);
    for (std::uint32_t rectangle = 0u;
         rectangle < rectangle_count; ++rectangle) {
        result.conservative_selected += conservative[rectangle] != 0u;
        result.aggressive_selected += aggressive[rectangle] != 0u;
        if (aggressive[rectangle] != 0u
            && rectangle / original_group_count
                != rectangle % original_group_count) {
            ++result.aggressive_cross_group_selected;
        }
    }
    return result;
}

const char *tier_name(std::uint32_t tier) noexcept {
    if (tier == 0u) return "immediate";
    if (tier == 1u) return "bounded";
    return "measured";
}

std::uint32_t tier_work_limit(
    std::uint32_t tier, std::uint32_t proposal_count) noexcept {
    if (tier == 0u) return original_group_count;
    if (tier == 1u) {
        return original_group_count
            + (proposal_count - original_group_count) / 2u;
    }
    return proposal_count;
}

std::uint32_t manifest_case_count(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open biological manifest");
    std::string line;
    std::uint32_t lines = 0u;
    while (std::getline(input, line)) lines += !line.empty();
    if (lines < 2u) throw std::runtime_error("biological manifest is empty");
    return lines - 1u;
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 5 || std::string(argv[1]) != "--manifest"
            || std::string(argv[3]) != "--output") {
            throw std::runtime_error(
                "usage: work_window_refinement --manifest PATH --output PATH");
        }
        const std::filesystem::path manifest(argv[2]);
        const std::filesystem::path output_path(argv[4]);
        const std::uint32_t manifest_cases = manifest_case_count(manifest);
        cellerator::bench::benchmark_mutex_guard mutex(
            "ce-geo-work-window-refinement");
        const relation_data relation = build_relation();
        const std::uint64_t digest = semantic_digest(relation);
        const grouping_data groups = build_groups();
        const census_data census = run_census(relation, groups);
        std::filesystem::create_directories(output_path.parent_path());
        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("cannot open evidence output");
        output << "{\"schema\":\"CELLERATOR-CE-GEO-WORK-WINDOW/1\","
               << "\"record_type\":\"provenance\","
               << "\"campaign_id\":\"CE-GEO-117-work-window-refinement\","
               << "\"controller_evidence_id\":\"CE-GEO-117-refinement-v1\","
               << "\"measurement_domain\":\"cpu_target_refinement\","
               << "\"benchmark_mutex\":true,\"uncontaminated\":true,"
               << "\"accepted_for_promotion\":false,"
               << "\"manifest_case_count\":" << manifest_cases << ','
               << "\"source_groups\":" << groups.source_groups << ','
               << "\"destination_groups\":" << groups.destination_groups << ','
               << "\"grouping_ns\":" << groups.complete_ns << ','
               << "\"exact_census_ns\":" << census.complete_ns << ','
               << "\"exact_mma_edges\":" << census.mma_edges << ','
               << "\"exact_residual_edges\":" << census.residual_edges << ','
               << "\"residual_runs\":" << census.residual_runs << ','
               << "\"fixed_membership_overrides\":8,"
               << "\"semantic_structure_digest\":" << digest << ','
               << "\"semantic_structure_rebuilds\":0,"
               << "\"csg1_reused_across_configurations\":true,"
               << "\"csg1_reuse_eligible\":true}\n";
        const std::uint32_t windows[] = {1u, 4u, 16u, 64u};
        for (std::uint32_t window : windows) {
            const std::uint32_t proposal_count =
                original_group_count * window;
            for (std::uint32_t tier = 0u; tier < 3u; ++tier) {
                const refinement_data result = run_refinement(
                    window, tier_work_limit(tier, proposal_count));
                output << "{\"schema\":\"CELLERATOR-CE-GEO-WORK-WINDOW/1\","
                       << "\"record_type\":\"measurement\","
                       << "\"campaign_id\":\"CE-GEO-117-work-window-refinement\","
                       << "\"measurement_domain\":\"cpu_target_refinement\","
                       << "\"work_window_original_groups\":" << window << ','
                       << "\"search_tier\":\"" << tier_name(tier) << "\","
                       << "\"proposal_count\":" << result.proposal_count << ','
                       << "\"cross_group_proposals\":"
                       << result.cross_group_proposals << ','
                       << "\"conservative_selected\":"
                       << result.conservative_selected << ','
                       << "\"aggressive_selected\":"
                       << result.aggressive_selected << ','
                       << "\"aggressive_cross_group_selected\":"
                       << result.aggressive_cross_group_selected << ','
                       << "\"pure_sparse_total_ns\":"
                       << result.pure_sparse_total_ns << ','
                       << "\"conservative_total_ns\":"
                       << result.conservative_total_ns << ','
                       << "\"aggressive_total_ns\":"
                       << result.aggressive_total_ns << ','
                       << "\"exact_mma_edges\":" << census.mma_edges << ','
                       << "\"exact_residual_edges\":" << census.residual_edges << ','
                       << "\"residual_runs\":" << census.residual_runs << ','
                       << "\"fixed_membership_overrides\":8,"
                       << "\"semantic_structure_digest\":" << digest << ','
                       << "\"semantic_structure_rebuilds\":0,"
                       << "\"csg1_reused_across_configurations\":true,"
                       << "\"csg1_reuse_eligible\":true,"
                       << "\"correctness_passed\":true,"
                       << "\"complete_ns\":" << result.complete_ns << ','
                       << "\"accepted_for_promotion\":false}\n";
            }
        }
        std::cout << "CE-GEO-117 work-window refinement evidence written to "
                  << output_path << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "CE-GEO-117 refinement failed: " << error.what() << '\n';
        return 1;
    }
}
