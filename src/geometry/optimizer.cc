// Proposal ordering and the snapshot/rollback controller intentionally share
// this TU: one private mutation key defines deterministic generation, replay,
// rejection, and diagnostics. Split only if that contract receives a stable
// private boundary; scattering it today would make ordering drift easier.
#include "Cellerator/geometry/optimizer.hh"

#include "optimizer_state.hh"

#include <Cellerator/geometry/gene_support_bitset.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cellpack {
namespace {

using clock_type = std::chrono::steady_clock;

constexpr u64 fnv1a_offset = 1469598103934665603ull;
constexpr u64 fnv1a_prime = 1099511628211ull;

void hash_byte(u64 *hash, unsigned char value) noexcept {
    *hash ^= value;
    *hash *= fnv1a_prime;
}

void hash_u64(u64 *hash, u64 value) noexcept {
    for (u32 byte = 0u; byte < 8u; ++byte) {
        hash_byte(hash, static_cast<unsigned char>(value >> (byte * 8u)));
    }
}

void hash_string(u64 *hash, const std::string &value) noexcept {
    hash_u64(hash, static_cast<u64>(value.size()));
    for (unsigned char byte : value) hash_byte(hash, byte);
}

template<class Values>
void hash_u64_values(u64 *hash, const Values &values) noexcept {
    hash_u64(hash, static_cast<u64>(values.size()));
    for (const auto value : values) hash_u64(hash, static_cast<u64>(value));
}

u64 sampled_support_identity_unchecked(const sampled_feature_support_view &support) noexcept {
    const auto &provenance = *support.provenance;
    u64 hash = fnv1a_offset;
    hash_string(&hash, "cellerator_sampled_feature_support_identity_v1");
    hash_u64(&hash, sampled_feature_support_identity_version);
    hash_u64(&hash, provenance.seed);
    hash_string(&hash, provenance.hash_algorithm);
    hash_u64(&hash, provenance.hash_version);
    hash_u64(&hash, provenance.total_rows);
    hash_u64(&hash, provenance.selected_rows);
    hash_u64(&hash, static_cast<u64>(provenance.mode));
    hash_string(&hash, provenance.split_name);
    hash_u64(&hash, static_cast<u64>(provenance.cell_identity));
    hash_u64(&hash, provenance.quantile.begin.numerator);
    hash_u64(&hash, provenance.quantile.begin.denominator);
    hash_u64(&hash, provenance.quantile.end.numerator);
    hash_u64(&hash, provenance.quantile.end.denominator);
    hash_u64(&hash, provenance.requested_row_count);
    hash_u64(&hash, provenance.requested_density_strata);
    hash_u64(&hash, provenance.density_strata);
    hash_u64_values(&hash, provenance.density_bin_upper_bounds_inclusive);
    hash_u64_values(&hash, provenance.stratum_total_rows);
    hash_u64_values(&hash, provenance.stratum_sampled_rows);
    hash_string(&hash, provenance.weighting_rule);
    hash_u64(&hash, support.sampled_row_count);
    for (u32 row = 0u; row < support.sampled_row_count; ++row) {
        hash_u64(&hash, support.sampled_position_to_global_row[row]);
    }
    return hash == 0u ? 1u : hash;
}

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

struct nominated_pair {
    u32 lhs = invalid_id;
    u32 rhs = invalid_id;
    candidate_relation evidence{};
};

class proposal_relation_workspace {
public:
    proposal_relation_workspace(u64 relation_count, u32 feature_count, const packing_optimizer_config &config)
        : fanout_(feature_count, 0u), block_marks_(feature_count, 0u),
          feature_marks_(feature_count, 0u) {
        const std::size_t relations = static_cast<std::size_t>(relation_count);
        pair_relations_.reserve(relations);
        ranked_relations_.reserve(relations);
        nominated_relations_.reserve(relations);
        raw_.reserve(static_cast<std::size_t>(config.proposal_shortlist) * 3u + 3u);
        proposals_.reserve(std::max<std::size_t>(relations, config.proposal_shortlist));
        selected_.reserve(config.initial_oracle_batch_size);
        blacklist_.reserve(config.maximum_oracle_evaluations);
    }

    bool blacklisted(const mutation_key &key) const noexcept {
        return std::find_if(blacklist_.begin(), blacklist_.end(), [&](const mutation_key &item) {
            return !(item < key) && !(key < item);
        }) != blacklist_.end();
    }
    void blacklist(const mutation_key &key) { blacklist_.push_back(key); }
    void reset_fanout() { std::fill(fanout_.begin(), fanout_.end(), 0u); }
    void next_marks() {
        if (++mark_generation_ == 0u) {
            std::fill(block_marks_.begin(), block_marks_.end(), 0u);
            std::fill(feature_marks_.begin(), feature_marks_.end(), 0u);
            mark_generation_ = 1u;
        }
    }
    bool block_marked(u32 value) const { return block_marks_[value] == mark_generation_; }
    bool feature_marked(u32 value) const { return feature_marks_[value] == mark_generation_; }
    void mark_block(u32 value) { block_marks_[value] = mark_generation_; }
    void mark_feature(u32 value) { feature_marks_[value] = mark_generation_; }
    std::size_t estimated_bytes() const noexcept {
        return pair_relations_.capacity() * sizeof(nominated_pair)
            + ranked_relations_.capacity() * sizeof(candidate_relation *)
            + nominated_relations_.capacity() * sizeof(candidate_relation *)
            + raw_.capacity() * sizeof(mutation_proposal)
            + proposals_.capacity() * sizeof(mutation_proposal)
            + selected_.capacity() * sizeof(mutation_proposal)
            + blacklist_.capacity() * sizeof(mutation_key)
            + (fanout_.capacity() + block_marks_.capacity() + feature_marks_.capacity()) * sizeof(u32);
    }

    std::vector<nominated_pair> pair_relations_;
    std::vector<const candidate_relation *> ranked_relations_;
    std::vector<const candidate_relation *> nominated_relations_;
    std::vector<mutation_proposal> raw_;
    std::vector<mutation_proposal> proposals_;
    std::vector<mutation_proposal> selected_;
    std::vector<mutation_key> blacklist_;
    std::vector<u32> fanout_;
    std::vector<u32> block_marks_;
    std::vector<u32> feature_marks_;
    u32 mark_generation_ = 0u;
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
    if (support.provenance != nullptr) {
        u64 actual_sampling_identity = 0u;
        const validation_result identity_status = query_sampled_feature_support_identity(
            support, &actual_sampling_identity);
        if (!identity_status) return identity_status;
        if (config.plan_identity.sampling_provenance_identity != actual_sampling_identity) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                "optimizer sampling provenance identity does not match sampled support");
        }
        if (config.plan_identity.row_domain_kind == packing_row_domain_kind::full_dataset_identity
            && source.support.row_count != support.provenance->total_rows) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                "full-domain optimizer evaluator row count does not match sampling provenance population");
        }
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

const std::vector<mutation_proposal> &generate_merge_proposals(
    const detail::optimizer_state &state,
    candidate_relation_view candidates,
    const packing_optimizer_config &config,
    proposal_relation_workspace *workspace,
    packing_optimizer_diagnostics *diagnostics) {
    std::vector<nominated_pair> &ranked = workspace->pair_relations_;
    ranked.clear();
    for (u64 index = 0u; index < candidates.relation_count; ++index) {
        const candidate_relation &relation = candidates.relations[index];
        u32 lhs = state.block_slot_for_feature(relation.feature_a);
        u32 rhs = state.block_slot_for_feature(relation.feature_b);
        if (lhs == rhs) continue;
        std::tie(lhs, rhs) = ordered_block_pair(state, lhs, rhs);
        if (state.block_width(lhs) + state.block_width(rhs) > config.maximum_feature_block_width) continue;
        ++diagnostics->merge_proposals_considered;
        ranked.push_back({lhs, rhs, relation});
    }
    std::sort(ranked.begin(), ranked.end(), [&](const nominated_pair &lhs, const nominated_pair &rhs) {
        const std::pair<u32, u32> lhs_key{state.block_stable_key(lhs.lhs), state.block_stable_key(lhs.rhs)};
        const std::pair<u32, u32> rhs_key{state.block_stable_key(rhs.lhs), state.block_stable_key(rhs.rhs)};
        if (lhs_key != rhs_key) return lhs_key < rhs_key;
        if (evidence_better(lhs.evidence, rhs.evidence)) return true;
        if (evidence_better(rhs.evidence, lhs.evidence)) return false;
        return false;
    });
    std::size_t compacted = 0u;
    for (std::size_t index = 0u; index < ranked.size();) {
        ranked[compacted++] = ranked[index];
        const u32 lhs_key = state.block_stable_key(ranked[index].lhs);
        const u32 rhs_key = state.block_stable_key(ranked[index].rhs);
        do { ++index; } while (index < ranked.size()
            && state.block_stable_key(ranked[index].lhs) == lhs_key
            && state.block_stable_key(ranked[index].rhs) == rhs_key);
    }
    ranked.resize(compacted);
    std::sort(ranked.begin(), ranked.end(), [&](const nominated_pair &lhs, const nominated_pair &rhs) {
        if (evidence_better(lhs.evidence, rhs.evidence)) return true;
        if (evidence_better(rhs.evidence, lhs.evidence)) return false;
        return std::pair<u32, u32>{state.block_stable_key(lhs.lhs), state.block_stable_key(lhs.rhs)}
            < std::pair<u32, u32>{state.block_stable_key(rhs.lhs), state.block_stable_key(rhs.rhs)};
    });
    workspace->reset_fanout();
    std::vector<mutation_proposal> &proposals = workspace->proposals_;
    proposals.clear();
    for (const nominated_pair &pair : ranked) {
        const u32 lhs_key = state.block_stable_key(pair.lhs), rhs_key = state.block_stable_key(pair.rhs);
        if (workspace->fanout_[lhs_key] >= config.candidate_fanout
            || workspace->fanout_[rhs_key] >= config.candidate_fanout) continue;
        ++workspace->fanout_[lhs_key];
        ++workspace->fanout_[rhs_key];
        mutation_proposal proposal;
        proposal.key = {mutation_kind::merge, invalid_id, invalid_id, lhs_key, rhs_key};
        if (workspace->blacklisted(proposal.key)) continue;
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

const std::vector<mutation_proposal> &generate_refinement_proposals(
    const detail::optimizer_state &state,
    candidate_relation_view candidates,
    const packing_optimizer_config &config,
    proposal_relation_workspace *workspace,
    packing_optimizer_diagnostics *diagnostics) {
    std::vector<const candidate_relation *> &ranked_relations = workspace->ranked_relations_;
    ranked_relations.clear();
    for (u64 index = 0u; index < candidates.relation_count; ++index) {
        ranked_relations.push_back(candidates.relations + index);
    }
    std::sort(ranked_relations.begin(), ranked_relations.end(), [](const candidate_relation *lhs, const candidate_relation *rhs) {
        if (evidence_better(*lhs, *rhs)) return true;
        if (evidence_better(*rhs, *lhs)) return false;
        return std::pair<u32, u32>{lhs->feature_a, lhs->feature_b}
            < std::pair<u32, u32>{rhs->feature_a, rhs->feature_b};
    });
    workspace->reset_fanout();
    std::vector<const candidate_relation *> &nominated_relations = workspace->nominated_relations_;
    nominated_relations.clear();
    for (const candidate_relation *relation : ranked_relations) {
        if (workspace->fanout_[relation->feature_a] >= config.candidate_fanout
            || workspace->fanout_[relation->feature_b] >= config.candidate_fanout) continue;
        ++workspace->fanout_[relation->feature_a];
        ++workspace->fanout_[relation->feature_b];
        nominated_relations.push_back(relation);
    }

    std::vector<mutation_proposal> &raw = workspace->raw_;
    raw.clear();
    auto compact_raw = [&]() {
        std::sort(raw.begin(), raw.end(), [&](const mutation_proposal &lhs, const mutation_proposal &rhs) {
            if (lhs.key < rhs.key) return true;
            if (rhs.key < lhs.key) return false;
            if (evidence_better(lhs.evidence, rhs.evidence)) return true;
            if (evidence_better(rhs.evidence, lhs.evidence)) return false;
            return false;
        });
        std::size_t compacted = 0u;
        for (std::size_t index = 0u; index < raw.size();) {
            raw[compacted++] = raw[index];
            const mutation_key key = raw[index].key;
            do { ++index; } while (index < raw.size()
                && !(raw[index].key < key) && !(key < raw[index].key));
        }
        raw.resize(compacted);
        std::sort(raw.begin(), raw.end(), [&](const mutation_proposal &lhs, const mutation_proposal &rhs) {
            if (evidence_better(lhs.evidence, rhs.evidence)) return true;
            if (evidence_better(rhs.evidence, lhs.evidence)) return false;
            return lhs.key < rhs.key;
        });
        if (raw.size() > config.proposal_shortlist) raw.resize(config.proposal_shortlist);
    };
    bool saturated = false;
    for (const candidate_relation *relation_pointer : nominated_relations) {
        const candidate_relation &relation = *relation_pointer;
        const u32 slot_a = state.block_slot_for_feature(relation.feature_a);
        const u32 slot_b = state.block_slot_for_feature(relation.feature_b);
        if (slot_a == slot_b) continue;
        if (saturated) {
            diagnostics->move_proposals_considered += config.enable_feature_moves ? 2u : 0u;
            diagnostics->swap_proposals_considered += config.enable_feature_swaps ? 1u : 0u;
            continue;
        }
        const u32 key_a = state.block_stable_key(slot_a), key_b = state.block_stable_key(slot_b);
        auto insert_raw = [&](mutation_proposal proposal) {
            if (!workspace->blacklisted(proposal.key)) raw.push_back(proposal);
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
        if (raw.size() >= static_cast<std::size_t>(config.proposal_shortlist) * 2u) {
            compact_raw();
            saturated = raw.size() >= config.proposal_shortlist;
        }
    }
    compact_raw();
    std::vector<mutation_proposal> &ranked = workspace->proposals_;
    ranked.assign(raw.begin(), raw.end());
    raw.clear();
    std::vector<mutation_proposal> &proposals = raw;
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

const std::vector<mutation_proposal> &select_batch(
    const std::vector<mutation_proposal> &proposals,
    u32 limit,
    proposal_relation_workspace *workspace) {
    workspace->next_marks();
    std::vector<mutation_proposal> &selected = workspace->selected_;
    selected.clear();
    for (const mutation_proposal &proposal : proposals) {
        if (selected.size() >= limit) break;
        if (workspace->block_marked(proposal.slot_a) || workspace->block_marked(proposal.slot_b)) continue;
        if (proposal.key.feature_a != invalid_id && workspace->feature_marked(proposal.key.feature_a)) continue;
        if (proposal.key.feature_b != invalid_id && workspace->feature_marked(proposal.key.feature_b)) continue;
        workspace->mark_block(proposal.slot_a);
        workspace->mark_block(proposal.slot_b);
        if (proposal.key.feature_a != invalid_id) workspace->mark_feature(proposal.key.feature_a);
        if (proposal.key.feature_b != invalid_id) workspace->mark_feature(proposal.key.feature_b);
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
    const validation_result journal_status = state->journal_blocks(proposal.slot_a, proposal.slot_b);
    if (!journal_status) return journal_status;
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
    proposal_relation_workspace *proposal_workspace,
    bool *accepted) {
    *accepted = false;
    u32 batch_limit = config.initial_oracle_batch_size;
    while (diagnostics->oracle_evaluations + 1u < config.maximum_oracle_evaluations) {
        const clock_type::time_point proxy_begin = clock_type::now();
        const std::vector<mutation_proposal> &proposals = generate();
        const clock_type::time_point proxy_end = clock_type::now();
        diagnostics->proxy_ms += milliseconds(proxy_begin, proxy_end);
        if (proposals.empty()) return validation_ok();
        const std::vector<mutation_proposal> &batch = select_batch(
            proposals, batch_limit, proposal_workspace);
        if (batch.empty()) return validation_ok();
        state->begin_mutation_journal();
        for (const mutation_proposal &proposal : batch) {
            const validation_result mutation_status = apply_mutation(state, proposal);
            if (!mutation_status) {
                state->rollback_mutation_journal();
                return mutation_status;
            }
        }
        evaluated_geometry evaluated;
        const validation_result evaluation_status = evaluate_state(
            state, source, config, workspace, &evaluated, diagnostics);
        if (!evaluation_status) {
            state->rollback_mutation_journal();
            return evaluation_status;
        }
        if (exact_improvement(evaluated.summary, *current, config)) {
            state->commit_mutation_journal();
            *current = evaluated.summary;
            record_accepts(batch, diagnostics);
            *accepted = true;
            return validation_ok();
        }
        const validation_result rollback_status = state->rollback_mutation_journal();
        if (!rollback_status) return rollback_status;
        ++diagnostics->oracle_rollbacks;
        record_rejects(batch, diagnostics);
        if (batch.size() > 1u) {
            batch_limit = std::max<u32>(1u, static_cast<u32>(batch.size() / 2u));
            ++diagnostics->oracle_batch_reductions;
        } else {
            proposal_workspace->blacklist(batch.front().key);
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
    result.provenance = source.provenance;
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
    if (support.provenance != nullptr) {
        if (support.provenance->selected_rows != support.sampled_row_count
            || support.provenance->total_rows < support.sampled_row_count) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                "sampled feature support provenance dimensions are inconsistent");
        }
        for (u32 row = 0u; row < support.sampled_row_count; ++row) {
            const u64 global_row = support.sampled_position_to_global_row[row];
            if (global_row >= support.provenance->total_rows
                || (row != 0u && global_row <= support.sampled_position_to_global_row[row - 1u])) {
                return validation_error(validation_code::invalid_plan_geometry, row,
                    "sampled/global row mapping is not canonical for its provenance");
            }
        }
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

validation_result query_sampled_feature_support_identity(
    const sampled_feature_support_view &support,
    u64 *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "sampled feature support identity output is null");
    }
    const validation_result status = validate_sampled_feature_support_view(support);
    if (!status) return status;
    if (support.provenance == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "sampled feature support provenance is unavailable for identity");
    }
    *out = sampled_support_identity_unchecked(support);
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
            config.maximum_feature_block_width, config.row_group_width,
            config.initial_oracle_batch_size);
        if (!status) return status;
        proposal_relation_workspace proposal_workspace(
            normalized.view().relation_count, sampled_support.feature_count, config);
        result.diagnostics.initial_block_count = state.active_block_count();
        result.diagnostics.peak_additional_optimizer_bytes = state.estimated_additional_bytes()
            + static_cast<std::size_t>(normalized.view().relation_count) * sizeof(candidate_relation)
            + proposal_workspace.estimated_bytes();

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
                [&]() {
                    return generate_merge_proposals(state, normalized.view(), config,
                        &proposal_workspace, &result.diagnostics);
                }, &current, &result.diagnostics, &proposal_workspace, &accepted);
            if (!status) return status;
            ++result.diagnostics.coarsening_passes;
            result.diagnostics.peak_additional_optimizer_bytes = std::max(
                result.diagnostics.peak_additional_optimizer_bytes,
                state.estimated_additional_bytes()
                    + static_cast<std::size_t>(normalized.view().relation_count) * sizeof(candidate_relation)
                    + proposal_workspace.estimated_bytes());
            if (!accepted) break;
        }

        result.diagnostics.final_phase = packing_optimizer_phase::refinement;
        for (u32 pass = 0u; pass < config.maximum_refinement_passes; ++pass) {
            bool accepted = false;
            status = run_one_accepted_batch(&state, source, config, workspace,
                [&]() {
                    return generate_refinement_proposals(state, normalized.view(), config,
                        &proposal_workspace, &result.diagnostics);
                }, &current, &result.diagnostics, &proposal_workspace, &accepted);
            if (!status) return status;
            ++result.diagnostics.refinement_passes;
            result.diagnostics.peak_additional_optimizer_bytes = std::max(
                result.diagnostics.peak_additional_optimizer_bytes,
                state.estimated_additional_bytes()
                    + static_cast<std::size_t>(normalized.view().relation_count) * sizeof(candidate_relation)
                    + proposal_workspace.estimated_bytes());
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
