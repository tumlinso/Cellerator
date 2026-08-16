// Proposal ordering and the snapshot/rollback controller intentionally share
// this TU: one private mutation key defines deterministic generation, replay,
// rejection, and diagnostics. Split only if that contract receives a stable
// private boundary; scattering it today would make ordering drift easier.
#include "CellPack/optimizer.hh"

#include "optimizer_state.hh"

#include <Cellerator/compute/gene_support_bitset.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace cellpack {
namespace {

using clock_type = std::chrono::steady_clock;

enum class mutation_kind : u32 { merge = 0u, move = 1u, swap = 2u };

struct mutation_key {
    mutation_kind kind = mutation_kind::merge;
    u32 feature_a = invalid_id;
    u32 feature_b = invalid_id;
    u32 block_a_key = invalid_id;
    u32 block_b_key = invalid_id;

    bool operator<(const mutation_key &rhs) const noexcept {
        return std::tie(kind, feature_a, feature_b, block_a_key, block_b_key)
            < std::tie(rhs.kind, rhs.feature_a, rhs.feature_b, rhs.block_a_key, rhs.block_b_key);
    }
};

struct mutation_proposal {
    mutation_key key{};
    u32 slot_a = invalid_id;
    u32 slot_b = invalid_id;
    u32 generation_a = 0u;
    u32 generation_b = 0u;
    std::int64_t proxy_gain = 0;
    candidate_relation evidence{};
    u32 combined_width = 0u;
};

struct evaluated_geometry {
    frozen_evaluation_summary summary{};
};

double milliseconds(clock_type::time_point begin, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

int evidence_rank(const candidate_relation &relation) noexcept {
    if ((relation.evidence_flags & candidate_evidence_exact) != 0u) return 2;
    if ((relation.evidence_flags & candidate_evidence_approximate) != 0u) return 1;
    return 0;
}

int compare_rational(const candidate_relation &lhs, const candidate_relation &rhs) noexcept {
    const __int128 left = static_cast<__int128>(lhs.score_numerator)
        * static_cast<__int128>(rhs.score_denominator);
    const __int128 right = static_cast<__int128>(rhs.score_numerator)
        * static_cast<__int128>(lhs.score_denominator);
    return left < right ? -1 : (left > right ? 1 : 0);
}

bool evidence_better(const candidate_relation &lhs, const candidate_relation &rhs) noexcept {
    const int lhs_rank = evidence_rank(lhs), rhs_rank = evidence_rank(rhs);
    if (lhs_rank != rhs_rank) return lhs_rank > rhs_rank;
    if (lhs.score_kind != rhs.score_kind) {
        return static_cast<u32>(lhs.score_kind) < static_cast<u32>(rhs.score_kind);
    }
    const int rational = compare_rational(lhs, rhs);
    if (rational != 0) return rational > 0;
    if (lhs.feature_a != rhs.feature_a) return lhs.feature_a < rhs.feature_a;
    return lhs.feature_b < rhs.feature_b;
}

bool proposal_better(const mutation_proposal &lhs, const mutation_proposal &rhs) noexcept {
    if (lhs.proxy_gain != rhs.proxy_gain) return lhs.proxy_gain > rhs.proxy_gain;
    if (evidence_better(lhs.evidence, rhs.evidence)) return true;
    if (evidence_better(rhs.evidence, lhs.evidence)) return false;
    if (lhs.combined_width != rhs.combined_width) return lhs.combined_width < rhs.combined_width;
    return lhs.key < rhs.key;
}

validation_result validate_optimizer_config(
    const prepared_csr_support &source,
    const sampled_feature_support_view &support,
    const packing_optimizer_config &config) {
    if (!source.validated) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id, "optimizer source has not been prepared");
    }
    if (source.support.feature_count != support.feature_count) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "sampled support and evaluator feature counts differ");
    }
    if (config.maximum_feature_block_width == 0u || config.row_group_width == 0u
        || config.candidate_fanout == 0u || config.proposal_shortlist == 0u
        || config.initial_oracle_batch_size == 0u || config.maximum_oracle_evaluations < 2u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "optimizer limits must be nonzero and reserve baseline/final oracle calls");
    }
    if (config.plan_identity.row_domain_kind == packing_row_domain_kind::unknown) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "optimizer row-domain identity must be explicit");
    }
    if (config.plan_identity.feature_axis_fingerprint == 0u
        || config.plan_identity.feature_axis_fingerprint_version == 0u
        || config.plan_identity.row_domain_identity == 0u
        || config.plan_identity.evaluation_source_identity == 0u
        || (config.plan_identity.row_domain_kind == packing_row_domain_kind::sampled_rows_identity
            && config.plan_identity.sampling_provenance_identity == 0u)
        || config.cost_policy_identity == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "optimizer compatibility/provenance identities and cost policy identity must be explicit");
    }
    const bool byte_geometry = config.cost_model.dense_values_within_occupied_tiles
        || config.cost_model.occupied_tile_metadata_bytes != 0u
        || config.cost_model.row_active_block_metadata_bytes != 0u;
    if (config.objective_kind == packing_exact_objective_kind::total_bytes && !byte_geometry) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "total-byte objective is geometry invariant under this cost model");
    }
    if (config.objective_kind == packing_exact_objective_kind::weighted_score
        && !byte_geometry
        && config.cost_model.occupied_tile_weight == 0.0
        && config.cost_model.row_active_block_weight == 0.0) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "weighted objective is geometry invariant under this cost model");
    }
    if (!std::isfinite(config.weighted_score_absolute_tolerance)
        || !std::isfinite(config.weighted_score_relative_tolerance)
        || config.weighted_score_absolute_tolerance < 0.0
        || config.weighted_score_relative_tolerance < 0.0) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "weighted score tolerances are invalid");
    }
    return validation_ok();
}

validation_result evaluate_state(
    detail::optimizer_state *state,
    const prepared_csr_support &source,
    const packing_optimizer_config &config,
    const packing_optimizer_workspace_view &workspace,
    evaluated_geometry *out,
    packing_optimizer_diagnostics *diagnostics) {
    validation_result status = state->materialize_execution_geometry();
    if (!status) return status;
    packing_plan_view plan;
    status = state->view(&plan);
    if (!status) return status;
    packing_occupancy_result occupancy;
    const clock_type::time_point begin = clock_type::now();
    status = evaluate_packing_plan(source, plan, workspace.evaluator_workspace,
        workspace.occupancy_buffers, &occupancy);
    if (!status) return status;
    packing_cost_estimate cost;
    status = estimate_packing_cost(occupancy, config.cost_model, &cost);
    const clock_type::time_point end = clock_type::now();
    if (!status) return status;
    diagnostics->oracle_ms += milliseconds(begin, end);
    ++diagnostics->oracle_evaluations;
    out->summary.occupancy = occupancy.totals;
    out->summary.cost = cost;
    if (config.objective_kind == packing_exact_objective_kind::total_bytes) {
        out->summary.objective = static_cast<double>(cost.total_bytes);
    } else if (config.objective_kind == packing_exact_objective_kind::row_active_block_references) {
        out->summary.objective = static_cast<double>(occupancy.totals.row_active_block_references);
    } else {
        out->summary.objective = cost.score;
    }
    return validation_ok();
}

bool exact_improvement(
    const frozen_evaluation_summary &candidate,
    const frozen_evaluation_summary &current,
    const packing_optimizer_config &config) noexcept {
    if (config.objective_kind == packing_exact_objective_kind::total_bytes) {
        return candidate.cost.total_bytes < current.cost.total_bytes;
    }
    if (config.objective_kind == packing_exact_objective_kind::row_active_block_references) {
        return candidate.occupancy.row_active_block_references
            < current.occupancy.row_active_block_references;
    }
    const double tolerance = config.weighted_score_absolute_tolerance
        + config.weighted_score_relative_tolerance
            * std::max(std::fabs(candidate.objective), std::fabs(current.objective));
    return candidate.objective < current.objective - tolerance;
}

std::pair<u32, u32> ordered_block_pair(const detail::optimizer_state &state, u32 lhs, u32 rhs) {
    if (state.block_stable_key(rhs) < state.block_stable_key(lhs)) std::swap(lhs, rhs);
    return {lhs, rhs};
}

std::vector<mutation_proposal> generate_merge_proposals(
    const detail::optimizer_state &state,
    candidate_relation_view candidates,
    const packing_optimizer_config &config,
    const std::set<mutation_key> &blacklist,
    packing_optimizer_diagnostics *diagnostics) {
    std::map<std::pair<u32, u32>, candidate_relation> best_by_pair;
    for (u64 index = 0u; index < candidates.relation_count; ++index) {
        const candidate_relation &relation = candidates.relations[index];
        u32 lhs = state.block_slot_for_feature(relation.feature_a);
        u32 rhs = state.block_slot_for_feature(relation.feature_b);
        if (lhs == rhs) continue;
        std::tie(lhs, rhs) = ordered_block_pair(state, lhs, rhs);
        if (state.block_width(lhs) + state.block_width(rhs) > config.maximum_feature_block_width) continue;
        ++diagnostics->merge_proposals_considered;
        const std::pair<u32, u32> key{state.block_stable_key(lhs), state.block_stable_key(rhs)};
        auto iterator = best_by_pair.find(key);
        if (iterator == best_by_pair.end() || evidence_better(relation, iterator->second)) {
            best_by_pair[key] = relation;
        }
    }

    struct nominated_pair { u32 lhs; u32 rhs; candidate_relation evidence; };
    std::vector<nominated_pair> ranked;
    ranked.reserve(best_by_pair.size());
    for (const auto &entry : best_by_pair) {
        const u32 lhs = state.block_slot_for_feature(entry.first.first);
        const u32 rhs = state.block_slot_for_feature(entry.first.second);
        if (lhs != rhs && state.block_active(lhs) && state.block_active(rhs)) ranked.push_back({lhs, rhs, entry.second});
    }
    std::sort(ranked.begin(), ranked.end(), [&](const nominated_pair &lhs, const nominated_pair &rhs) {
        if (evidence_better(lhs.evidence, rhs.evidence)) return true;
        if (evidence_better(rhs.evidence, lhs.evidence)) return false;
        return std::pair<u32, u32>{state.block_stable_key(lhs.lhs), state.block_stable_key(lhs.rhs)}
            < std::pair<u32, u32>{state.block_stable_key(rhs.lhs), state.block_stable_key(rhs.rhs)};
    });
    std::map<u32, u32> fanout;
    std::vector<mutation_proposal> proposals;
    for (const nominated_pair &pair : ranked) {
        const u32 lhs_key = state.block_stable_key(pair.lhs), rhs_key = state.block_stable_key(pair.rhs);
        if (fanout[lhs_key] >= config.candidate_fanout || fanout[rhs_key] >= config.candidate_fanout) continue;
        ++fanout[lhs_key];
        ++fanout[rhs_key];
        mutation_proposal proposal;
        proposal.key = {mutation_kind::merge, invalid_id, invalid_id, lhs_key, rhs_key};
        if (blacklist.count(proposal.key) != 0u) continue;
        proposal.slot_a = pair.lhs;
        proposal.slot_b = pair.rhs;
        proposal.generation_a = state.block_generation(pair.lhs);
        proposal.generation_b = state.block_generation(pair.rhs);
        proposal.proxy_gain = state.merge_proxy_gain(pair.lhs, pair.rhs);
        proposal.evidence = pair.evidence;
        proposal.combined_width = state.block_width(pair.lhs) + state.block_width(pair.rhs);
        ++diagnostics->merge_proposals_shortlisted;
        if (proposal.proxy_gain > 0) {
            ++diagnostics->merge_proxy_positive;
            proposals.push_back(proposal);
        }
    }
    std::sort(proposals.begin(), proposals.end(), proposal_better);
    if (proposals.size() > config.proposal_shortlist) proposals.resize(config.proposal_shortlist);
    return proposals;
}

std::vector<mutation_proposal> generate_refinement_proposals(
    const detail::optimizer_state &state,
    candidate_relation_view candidates,
    const packing_optimizer_config &config,
    const std::set<mutation_key> &blacklist,
    packing_optimizer_diagnostics *diagnostics) {
    std::vector<const candidate_relation *> ranked_relations;
    ranked_relations.reserve(static_cast<std::size_t>(candidates.relation_count));
    for (u64 index = 0u; index < candidates.relation_count; ++index) {
        ranked_relations.push_back(candidates.relations + index);
    }
    std::sort(ranked_relations.begin(), ranked_relations.end(), [](const candidate_relation *lhs, const candidate_relation *rhs) {
        if (evidence_better(*lhs, *rhs)) return true;
        if (evidence_better(*rhs, *lhs)) return false;
        return std::pair<u32, u32>{lhs->feature_a, lhs->feature_b}
            < std::pair<u32, u32>{rhs->feature_a, rhs->feature_b};
    });
    std::map<u32, u32> feature_fanout;
    std::vector<const candidate_relation *> nominated_relations;
    nominated_relations.reserve(ranked_relations.size());
    for (const candidate_relation *relation : ranked_relations) {
        if (feature_fanout[relation->feature_a] >= config.candidate_fanout
            || feature_fanout[relation->feature_b] >= config.candidate_fanout) continue;
        ++feature_fanout[relation->feature_a];
        ++feature_fanout[relation->feature_b];
        nominated_relations.push_back(relation);
    }

    std::map<mutation_key, mutation_proposal> raw;
    for (const candidate_relation *relation_pointer : nominated_relations) {
        const candidate_relation &relation = *relation_pointer;
        const u32 slot_a = state.block_slot_for_feature(relation.feature_a);
        const u32 slot_b = state.block_slot_for_feature(relation.feature_b);
        if (slot_a == slot_b) continue;
        const u32 key_a = state.block_stable_key(slot_a), key_b = state.block_stable_key(slot_b);
        auto insert_raw = [&](mutation_proposal proposal) {
            if (blacklist.count(proposal.key) != 0u) return;
            auto iterator = raw.find(proposal.key);
            if (iterator == raw.end() || evidence_better(proposal.evidence, iterator->second.evidence)) {
                raw[proposal.key] = proposal;
            }
        };
        if (config.enable_feature_moves) {
            ++diagnostics->move_proposals_considered;
            if (state.block_width(slot_b) < config.maximum_feature_block_width) {
                mutation_proposal move;
                move.key = {mutation_kind::move, relation.feature_a, invalid_id, key_a, key_b};
                move.slot_a = slot_a; move.slot_b = slot_b;
                move.generation_a = state.block_generation(slot_a); move.generation_b = state.block_generation(slot_b);
                move.evidence = relation;
                move.combined_width = state.block_width(slot_a) + state.block_width(slot_b);
                insert_raw(move);
            }
            ++diagnostics->move_proposals_considered;
            if (state.block_width(slot_a) < config.maximum_feature_block_width) {
                mutation_proposal move;
                move.key = {mutation_kind::move, relation.feature_b, invalid_id, key_b, key_a};
                move.slot_a = slot_b; move.slot_b = slot_a;
                move.generation_a = state.block_generation(slot_b); move.generation_b = state.block_generation(slot_a);
                move.evidence = relation;
                move.combined_width = state.block_width(slot_a) + state.block_width(slot_b);
                insert_raw(move);
            }
        }
        if (config.enable_feature_swaps) {
            ++diagnostics->swap_proposals_considered;
            mutation_proposal swap;
            const u32 first = std::min(relation.feature_a, relation.feature_b);
            const u32 second = std::max(relation.feature_a, relation.feature_b);
            swap.key = {mutation_kind::swap, first, second, std::min(key_a, key_b), std::max(key_a, key_b)};
            swap.slot_a = slot_a; swap.slot_b = slot_b;
            swap.generation_a = state.block_generation(slot_a); swap.generation_b = state.block_generation(slot_b);
            swap.evidence = relation;
            swap.combined_width = state.block_width(slot_a) + state.block_width(slot_b);
            insert_raw(swap);
        }
    }
    std::vector<mutation_proposal> ranked;
    ranked.reserve(raw.size());
    for (auto &entry : raw) ranked.push_back(entry.second);
    std::sort(ranked.begin(), ranked.end(), [&](const mutation_proposal &lhs, const mutation_proposal &rhs) {
        if (evidence_better(lhs.evidence, rhs.evidence)) return true;
        if (evidence_better(rhs.evidence, lhs.evidence)) return false;
        return lhs.key < rhs.key;
    });
    if (ranked.size() > config.proposal_shortlist) ranked.resize(config.proposal_shortlist);
    std::vector<mutation_proposal> proposals;
    proposals.reserve(ranked.size());
    for (mutation_proposal proposal : ranked) {
        if (proposal.key.kind == mutation_kind::move) {
            ++diagnostics->move_proposals_shortlisted;
            proposal.proxy_gain = state.move_proxy_gain(proposal.key.feature_a, proposal.slot_b);
            if (proposal.proxy_gain > 0) {
                ++diagnostics->move_proxy_positive;
                proposals.push_back(proposal);
            }
        } else {
            ++diagnostics->swap_proposals_shortlisted;
            proposal.proxy_gain = state.swap_proxy_gain(proposal.key.feature_a, proposal.key.feature_b);
            if (proposal.proxy_gain > 0) {
                ++diagnostics->swap_proxy_positive;
                proposals.push_back(proposal);
            }
        }
    }
    std::sort(proposals.begin(), proposals.end(), proposal_better);
    return proposals;
}

std::vector<mutation_proposal> select_batch(
    const std::vector<mutation_proposal> &proposals,
    u32 limit) {
    std::set<u32> used_blocks;
    std::set<u32> used_features;
    std::vector<mutation_proposal> selected;
    for (const mutation_proposal &proposal : proposals) {
        if (selected.size() >= limit) break;
        if (used_blocks.count(proposal.slot_a) != 0u || used_blocks.count(proposal.slot_b) != 0u) continue;
        if (proposal.key.feature_a != invalid_id && used_features.count(proposal.key.feature_a) != 0u) continue;
        if (proposal.key.feature_b != invalid_id && used_features.count(proposal.key.feature_b) != 0u) continue;
        used_blocks.insert(proposal.slot_a);
        used_blocks.insert(proposal.slot_b);
        if (proposal.key.feature_a != invalid_id) used_features.insert(proposal.key.feature_a);
        if (proposal.key.feature_b != invalid_id) used_features.insert(proposal.key.feature_b);
        selected.push_back(proposal);
    }
    return selected;
}

validation_result apply_mutation(detail::optimizer_state *state, const mutation_proposal &proposal) {
    if (!state->block_active(proposal.slot_a) || !state->block_active(proposal.slot_b)
        || state->block_generation(proposal.slot_a) != proposal.generation_a
        || state->block_generation(proposal.slot_b) != proposal.generation_b) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "stale optimizer mutation proposal");
    }
    if (proposal.key.kind == mutation_kind::merge) return state->merge_blocks(proposal.slot_a, proposal.slot_b);
    if (proposal.key.kind == mutation_kind::move) return state->move_feature(proposal.key.feature_a, proposal.slot_b);
    return state->swap_features(proposal.key.feature_a, proposal.key.feature_b);
}

void record_accepts(const std::vector<mutation_proposal> &batch, packing_optimizer_diagnostics *diagnostics) {
    for (const mutation_proposal &proposal : batch) {
        if (proposal.key.kind == mutation_kind::merge) ++diagnostics->merge_oracle_accepted;
        else if (proposal.key.kind == mutation_kind::move) ++diagnostics->move_oracle_accepted;
        else ++diagnostics->swap_oracle_accepted;
    }
}

void record_rejects(const std::vector<mutation_proposal> &batch, packing_optimizer_diagnostics *diagnostics) {
    for (const mutation_proposal &proposal : batch) {
        if (proposal.key.kind == mutation_kind::merge) ++diagnostics->merge_oracle_rejected;
        else if (proposal.key.kind == mutation_kind::move) ++diagnostics->move_oracle_rejected;
        else ++diagnostics->swap_oracle_rejected;
    }
}

template <class Generator>
validation_result run_one_accepted_batch(
    detail::optimizer_state *state,
    const prepared_csr_support &source,
    const packing_optimizer_config &config,
    const packing_optimizer_workspace_view &workspace,
    Generator generate,
    frozen_evaluation_summary *current,
    packing_optimizer_diagnostics *diagnostics,
    bool *accepted) {
    *accepted = false;
    std::set<mutation_key> blacklist;
    u32 batch_limit = config.initial_oracle_batch_size;
    while (diagnostics->oracle_evaluations + 1u < config.maximum_oracle_evaluations) {
        const clock_type::time_point proxy_begin = clock_type::now();
        std::vector<mutation_proposal> proposals = generate(blacklist);
        const clock_type::time_point proxy_end = clock_type::now();
        diagnostics->proxy_ms += milliseconds(proxy_begin, proxy_end);
        if (proposals.empty()) return validation_ok();
        std::vector<mutation_proposal> batch = select_batch(proposals, batch_limit);
        if (batch.empty()) return validation_ok();
        detail::optimizer_state snapshot = *state;
        for (const mutation_proposal &proposal : batch) {
            const validation_result mutation_status = apply_mutation(state, proposal);
            if (!mutation_status) return mutation_status;
        }
        evaluated_geometry evaluated;
        const validation_result evaluation_status = evaluate_state(
            state, source, config, workspace, &evaluated, diagnostics);
        if (!evaluation_status) return evaluation_status;
        if (exact_improvement(evaluated.summary, *current, config)) {
            *current = evaluated.summary;
            record_accepts(batch, diagnostics);
            *accepted = true;
            return validation_ok();
        }
        *state = std::move(snapshot);
        ++diagnostics->oracle_rollbacks;
        record_rejects(batch, diagnostics);
        if (batch.size() > 1u) {
            batch_limit = std::max<u32>(1u, static_cast<u32>(batch.size() / 2u));
            ++diagnostics->oracle_batch_reductions;
        } else {
            blacklist.insert(batch.front().key);
            ++diagnostics->blacklisted_mutations;
            batch_limit = config.initial_oracle_batch_size;
        }
    }
    return validation_ok();
}

} // namespace

validation_result make_sampled_feature_support_view(
    const ::cellerator::compute::gene_support::gene_support_bitset_view &source,
    sampled_feature_support_view *out) {
    if (out == nullptr) return validation_error(validation_code::null_pointer, invalid_id, "sampled feature support output is null");
    if (source.layout.sampled_cell_count > std::numeric_limits<u32>::max()
        || source.layout.gene_count > std::numeric_limits<u32>::max()
        || source.layout.words_per_gene > std::numeric_limits<u32>::max()) {
        return validation_error(validation_code::integer_overflow, invalid_id, "gene support dimensions exceed CP-BP-04 uint32 limits");
    }
    static_assert(sizeof(::cellerator::types::count_value_t) == sizeof(u32), "gene detection counts must remain uint32-compatible");
    sampled_feature_support_view result;
    result.sampled_row_count = static_cast<u32>(source.layout.sampled_cell_count);
    result.feature_count = static_cast<u32>(source.layout.gene_count);
    result.words_per_feature = static_cast<u32>(source.layout.words_per_gene);
    result.support_words = reinterpret_cast<const u32 *>(source.gene_support);
    result.detected_row_counts = reinterpret_cast<const u32 *>(source.detected_cell_counts);
    result.sampled_position_to_global_row = source.sampled_position_to_global_row;
    const validation_result status = validate_sampled_feature_support_view(result);
    if (!status) return status;
    *out = result;
    return validation_ok();
}

validation_result validate_sampled_feature_support_view(const sampled_feature_support_view &support) {
    const u32 expected_words = support.sampled_row_count == 0u
        ? 0u : 1u + ((support.sampled_row_count - 1u) / 32u);
    if (support.feature_count == 0u || support.words_per_feature != expected_words) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "sampled feature support dimensions are invalid");
    }
    if (support.words_per_feature != 0u && support.support_words == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "sampled feature support words are null");
    }
    if (support.sampled_row_count != 0u && support.sampled_position_to_global_row == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "sampled/global row mapping is null");
    }
    if (support.detected_row_counts != nullptr) {
        for (u32 feature = 0u; feature < support.feature_count; ++feature) {
            if (support.detected_row_counts[feature] > support.sampled_row_count) {
                return validation_error(validation_code::invalid_plan_geometry, feature, "sampled feature detection count exceeds sampled rows");
            }
        }
    }
    return validation_ok();
}

validation_result query_packing_optimizer_workspace_requirements(
    const prepared_csr_support &source,
    u32 row_group_width,
    packing_optimizer_workspace_requirements *out) {
    if (out == nullptr) return validation_error(validation_code::null_pointer, invalid_id, "optimizer requirements output is null");
    if (!source.validated || row_group_width == 0u || source.support.feature_count == 0u || source.support.row_count == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "optimizer requirements source or row-group width is invalid");
    }
    std::vector<u32> feature_offsets(static_cast<std::size_t>(source.support.feature_count) + 1u);
    for (u32 feature = 0u; feature <= source.support.feature_count; ++feature) feature_offsets[feature] = feature;
    std::vector<u32> row_offsets(1u, 0u);
    for (u64 boundary = row_group_width; boundary < source.support.row_count; boundary += row_group_width) {
        row_offsets.push_back(static_cast<u32>(boundary));
    }
    row_offsets.push_back(source.support.row_count);
    packing_plan_view singleton;
    singleton.row_count = source.support.row_count;
    singleton.feature_count = source.support.feature_count;
    singleton.row_group_count = static_cast<u32>(row_offsets.size() - 1u);
    singleton.row_group_offsets = row_offsets.data();
    singleton.feature_block_count = source.support.feature_count;
    singleton.feature_block_offsets = feature_offsets.data();
    packing_optimizer_workspace_requirements result;
    const validation_result status = query_packing_evaluation_requirements(source, singleton, &result.evaluator);
    if (!status) return status;
    *out = result;
    return validation_ok();
}

validation_result optimize_packing_plan(
    const prepared_csr_support &source,
    const sampled_feature_support_view &sampled_support,
    candidate_relation_view candidates,
    const packing_optimizer_config &config,
    const packing_optimizer_workspace_view &workspace,
    packing_optimizer_result *out) {
    if (out == nullptr) return validation_error(validation_code::null_pointer, invalid_id, "optimizer result is null");
    const clock_type::time_point total_begin = clock_type::now();
    const validation_result support_status = validate_sampled_feature_support_view(sampled_support);
    if (!support_status) return support_status;
    const validation_result config_status = validate_optimizer_config(source, sampled_support, config);
    if (!config_status) return config_status;
    try {
        packing_optimizer_result result;
        result.diagnostics.final_phase = packing_optimizer_phase::candidate_normalization;
        const clock_type::time_point candidate_begin = clock_type::now();
        normalized_candidate_relations normalized;
        validation_result status = normalize_candidate_relations(candidates, sampled_support.feature_count, &normalized);
        const clock_type::time_point candidate_end = clock_type::now();
        if (!status) return status;
        result.diagnostics.candidate_processing_ms = milliseconds(candidate_begin, candidate_end);
        result.diagnostics.candidate_normalization = normalized.statistics();

        detail::optimizer_state state;
        status = state.initialize(source.support.row_count, sampled_support,
            config.maximum_feature_block_width, config.row_group_width);
        if (!status) return status;
        result.diagnostics.initial_block_count = state.active_block_count();
        result.diagnostics.peak_additional_optimizer_bytes = 2u * state.estimated_additional_bytes()
            + static_cast<std::size_t>(normalized.view().relation_count) * sizeof(candidate_relation)
            + static_cast<std::size_t>(config.proposal_shortlist) * sizeof(mutation_proposal);

        result.diagnostics.final_phase = packing_optimizer_phase::baseline;
        evaluated_geometry baseline;
        status = evaluate_state(&state, source, config, workspace, &baseline, &result.diagnostics);
        if (!status) return status;
        frozen_evaluation_summary current = baseline.summary;
        result.diagnostics.baseline = baseline.summary;

        result.diagnostics.final_phase = packing_optimizer_phase::coarsening;
        for (u32 pass = 0u; pass < config.maximum_coarsening_passes; ++pass) {
            bool accepted = false;
            status = run_one_accepted_batch(&state, source, config, workspace,
                [&](const std::set<mutation_key> &blacklist) {
                    return generate_merge_proposals(state, normalized.view(), config, blacklist, &result.diagnostics);
                }, &current, &result.diagnostics, &accepted);
            if (!status) return status;
            ++result.diagnostics.coarsening_passes;
            result.diagnostics.peak_additional_optimizer_bytes = std::max(
                result.diagnostics.peak_additional_optimizer_bytes,
                2u * state.estimated_additional_bytes()
                    + static_cast<std::size_t>(normalized.view().relation_count) * sizeof(candidate_relation)
                    + static_cast<std::size_t>(config.proposal_shortlist) * sizeof(mutation_proposal));
            if (!accepted) break;
        }

        result.diagnostics.final_phase = packing_optimizer_phase::refinement;
        for (u32 pass = 0u; pass < config.maximum_refinement_passes; ++pass) {
            bool accepted = false;
            status = run_one_accepted_batch(&state, source, config, workspace,
                [&](const std::set<mutation_key> &blacklist) {
                    return generate_refinement_proposals(state, normalized.view(), config, blacklist, &result.diagnostics);
                }, &current, &result.diagnostics, &accepted);
            if (!status) return status;
            ++result.diagnostics.refinement_passes;
            result.diagnostics.peak_additional_optimizer_bytes = std::max(
                result.diagnostics.peak_additional_optimizer_bytes,
                2u * state.estimated_additional_bytes()
                    + static_cast<std::size_t>(normalized.view().relation_count) * sizeof(candidate_relation)
                    + static_cast<std::size_t>(config.proposal_shortlist) * sizeof(mutation_proposal));
            if (!accepted) break;
        }

        result.diagnostics.final_phase = packing_optimizer_phase::final_verification;
        evaluated_geometry final_evaluation;
        status = evaluate_state(&state, source, config, workspace, &final_evaluation, &result.diagnostics);
        if (!status) return status;
        if (exact_improvement(current, final_evaluation.summary, config)
            || exact_improvement(final_evaluation.summary, current, config)) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id, "final oracle result disagrees with accepted checkpoint");
        }
        if (exact_improvement(baseline.summary, final_evaluation.summary, config)) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id, "optimizer final exact objective regressed from baseline");
        }
        result.diagnostics.final = final_evaluation.summary;
        result.diagnostics.final_block_count = state.active_block_count();

        result.diagnostics.final_phase = packing_optimizer_phase::freeze;
        const clock_type::time_point freeze_begin = clock_type::now();
        frozen_packing_plan_build_view build;
        build.row_count = state.row_count();
        build.feature_count = state.feature_count();
        build.feature_permutation = state.feature_permutation().data();
        build.inverse_feature_permutation = state.inverse_feature_permutation().data();
        build.feature_block_count = static_cast<u32>(state.feature_block_offsets().size() - 1u);
        build.feature_block_offsets = state.feature_block_offsets().data();
        build.feature_to_block = state.feature_to_block().data();
        build.feature_to_local = state.feature_to_local().data();
        build.row_group_count = static_cast<u32>(state.row_group_offsets().size() - 1u);
        build.row_group_offsets = state.row_group_offsets().data();
        build.maximum_feature_block_width = config.maximum_feature_block_width;
        build.row_group_width = config.row_group_width;
        build.identity = config.plan_identity;
        build.objective_kind = config.objective_kind;
        build.cost_policy_identity = config.cost_policy_identity;
        build.baseline = baseline.summary;
        build.final = final_evaluation.summary;
        status = freeze_packing_plan(build, &result.plan);
        const clock_type::time_point freeze_end = clock_type::now();
        if (!status) return status;
        result.diagnostics.freeze_ms = milliseconds(freeze_begin, freeze_end);
        result.diagnostics.total_ms = milliseconds(total_begin, clock_type::now());
        *out = std::move(result);
        return validation_ok();
    } catch (const std::bad_alloc &) {
        return validation_error(validation_code::integer_overflow, invalid_id, "optimizer host allocation failed");
    }
}

} // namespace cellpack
