#include <Cellerator/compute/gating/condition_program.hh>
#include <Cellerator/compute/indexing/reverse_postings.hh>
#include <Cellerator/compute/support/logical_support.hh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace cellerator::compute;
using clock_type = std::chrono::steady_clock;

struct options {
    std::uint64_t edges = 2176;
    std::uint64_t cells = 64;
    std::uint64_t entities = 256;
    std::uint32_t query_terms = 1;
    double gate_density = 0.5;
    double skew = 0.0;
    double dirty_fraction = 0.01;
    std::uint64_t seed = 20261027;
    std::uint32_t samples = 21;
    std::uint64_t target_edge_visits = 2000000;
    bool all_terms = false;
    bool skip_worksets = false;
};

std::uint64_t splitmix64(std::uint64_t &state) {
    state += 0x9e3779b97f4a7c15ull;
    std::uint64_t x = state;
    x = (x ^ (x >> 30u)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27u)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31u);
}

std::uint64_t hash_mix(std::uint64_t h, std::uint64_t value) {
    value ^= value >> 30u;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27u;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31u;
    h ^= value + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
    return h;
}

double parse_double(const char *text, const char *name) {
    char *end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value))
        throw std::runtime_error(std::string("invalid ") + name);
    return value;
}

std::uint64_t parse_u64(const char *text, const char *name) {
    char *end = nullptr;
    const auto value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0')
        throw std::runtime_error(std::string("invalid ") + name);
    return static_cast<std::uint64_t>(value);
}

options parse_options(int argc, char **argv) {
    options result;
    const auto need_value = [&](int &i, const char *name) -> const char * {
        if (i + 1 >= argc)
            throw std::runtime_error(std::string("missing value for ") + name);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--edges") result.edges = parse_u64(need_value(i, "--edges"), "edges");
        else if (arg == "--cells") result.cells = parse_u64(need_value(i, "--cells"), "cells");
        else if (arg == "--entities") result.entities = parse_u64(need_value(i, "--entities"), "entities");
        else if (arg == "--query-terms") result.query_terms = static_cast<std::uint32_t>(
            parse_u64(need_value(i, "--query-terms"), "query terms"));
        else if (arg == "--gate-density") result.gate_density = parse_double(
            need_value(i, "--gate-density"), "gate density");
        else if (arg == "--skew") result.skew = parse_double(need_value(i, "--skew"), "skew");
        else if (arg == "--dirty-fraction") result.dirty_fraction = parse_double(
            need_value(i, "--dirty-fraction"), "dirty fraction");
        else if (arg == "--seed") result.seed = parse_u64(need_value(i, "--seed"), "seed");
        else if (arg == "--samples") result.samples = static_cast<std::uint32_t>(
            parse_u64(need_value(i, "--samples"), "samples"));
        else if (arg == "--target-edge-visits") result.target_edge_visits =
            parse_u64(need_value(i, "--target-edge-visits"), "target edge visits");
        else if (arg == "--match") {
            const std::string mode = need_value(i, "--match");
            if (mode == "all") result.all_terms = true;
            else if (mode == "any") result.all_terms = false;
            else throw std::runtime_error("--match must be any or all");
        } else if (arg == "--skip-worksets") result.skip_worksets = true;
        else if (arg == "--help") {
            std::cout << "sparse_support_index_bench [--edges N] [--cells N] "
                         "[--entities N] [--query-terms 1..3] "
                         "[--match any|all] [--gate-density 0..1] [--skew 0..1] "
                         "[--dirty-fraction 0..1] [--seed N] [--samples N] "
                         "[--target-edge-visits N] [--skip-worksets]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown option: " + arg);
    }
    if (result.edges == 0 || result.cells == 0 || result.entities < 3
        || result.query_terms == 0 || result.query_terms > 3
        || result.samples < 3 || result.target_edge_visits == 0
        || result.gate_density < 0.0 || result.gate_density > 1.0
        || result.skew < 0.0 || result.skew > 1.0
        || result.dirty_fraction < 0.0 || result.dirty_fraction > 1.0)
        throw std::runtime_error("shape and probability values are outside supported bounds");
    return result;
}

struct relation_storage {
    std::vector<support::logical_support_edge_v1> edges;
    std::vector<support::logical_support_participant_v1> participants;
    support::logical_support_view_v1 view{};
};

struct fixture {
    relation_storage base;
    relation_storage delta;
    support::support_snapshot_pair_v1 snapshot{};
    std::uint64_t hash = 1469598103934665603ull;
    std::uint64_t edge_count = 0;
    std::uint64_t participant_count = 0;
};

std::uint64_t choose_entity(std::uint64_t &rng, std::uint64_t entity_count,
                            double skew) {
    if (static_cast<double>(splitmix64(rng) >> 11u)
            / static_cast<double>(std::uint64_t{1} << 53u) < skew)
        return 1u;
    return 1u + splitmix64(rng) % entity_count;
}

fixture make_fixture(const options &cfg) {
    fixture result;
    const std::uint64_t delta_count = cfg.edges > 1
        ? std::max<std::uint64_t>(1u, cfg.edges / 16u) : 0u;
    const std::uint64_t base_count = cfg.edges - delta_count;
    result.edge_count = cfg.edges;
    std::uint64_t rng = cfg.seed;
    auto fill = [&](relation_storage &relation, std::uint64_t begin,
                    std::uint64_t count, std::uint64_t relation_id) {
        relation.edges.reserve(count);
        relation.participants.reserve(count * 4u);
        for (std::uint64_t local = 0; local < count; ++local) {
            const std::uint64_t ordinal = begin + local;
            const auto source = choose_entity(rng, cfg.entities, cfg.skew);
            const auto target = 1u + splitmix64(rng) % cfg.entities;
            const auto cofactor = 1u + splitmix64(rng) % cfg.entities;
            const auto participant_begin = relation.participants.size();
            relation.participants.push_back({0u, 0u, source});
            relation.participants.push_back({1u, 0u, target});
            relation.participants.push_back({2u, 0u, cofactor});
            // Repeated incidences are legal in logical support and must not
            // duplicate a role/entity/edge posting or candidate edge.
            const std::uint32_t participant_count = ordinal % 31u == 0u ? 4u : 3u;
            if (participant_count == 4u)
                relation.participants.push_back({0u, 0u, source});
            relation.edges.push_back({ordinal + 1u, participant_begin,
                                      participant_count, 0u});
            result.hash = hash_mix(result.hash, ordinal + 1u);
            result.hash = hash_mix(result.hash, source);
            result.hash = hash_mix(result.hash, target);
            result.hash = hash_mix(result.hash, cofactor);
            result.hash = hash_mix(result.hash, participant_count);
        }
        relation.view = {support::logical_support_schema_version_v1, 0u,
            relation_id, 9001u, 1u, relation.edges.data(),
            relation.edges.size(), relation.participants.data(),
            relation.participants.size()};
    };
    fill(result.base, 0u, base_count, 7001u);
    if (delta_count != 0u) fill(result.delta, base_count, delta_count, 7002u);
    result.participant_count = result.base.participants.size()
        + result.delta.participants.size();
    result.snapshot = {&result.base.view,
        delta_count == 0u ? nullptr : &result.delta.view, 1u};
    if (support::validate_support_snapshot_pair_v1(result.snapshot)
        != support::support_snapshot_status_v1::success)
        throw std::runtime_error("generated support fixture failed validation");
    return result;
}

struct term_batch {
    std::vector<indexing::posting_term_v1> primary;
    std::vector<indexing::posting_term_v1> derivative;
    std::uint64_t hash = 1469598103934665603ull;
};

bool term_less(const indexing::posting_term_v1 &a,
               const indexing::posting_term_v1 &b) {
    if (a.role != b.role) return a.role < b.role;
    return a.entity_id < b.entity_id;
}

term_batch make_terms(const options &cfg, const fixture &support_fixture) {
    term_batch result;
    result.primary.reserve(cfg.cells * cfg.query_terms);
    result.derivative.reserve(cfg.cells * cfg.query_terms);
    std::uint64_t rng = cfg.seed ^ 0xd1b54a32d192ed03ull;
    auto append_any = [&](std::vector<indexing::posting_term_v1> &terms,
                          std::uint64_t cell, std::uint32_t role) {
        std::vector<indexing::posting_term_v1> row;
        row.reserve(cfg.query_terms);
        for (std::uint32_t k = 0; k < cfg.query_terms; ++k) {
            for (;;) {
                const bool hot_source = role == 0u && cfg.skew > 0.5 && cell % 4u == 0u;
                const std::uint64_t entity = hot_source
                    ? 1u : 1u + splitmix64(rng) % cfg.entities;
                const auto exists = std::find_if(row.begin(), row.end(),
                    [&](const auto &item) { return item.entity_id == entity; });
                if (exists == row.end()) {
                    row.push_back({role, 0u, entity});
                    break;
                }
            }
        }
        std::sort(row.begin(), row.end(), term_less);
        terms.insert(terms.end(), row.begin(), row.end());
    };
    auto append_all = [&](std::vector<indexing::posting_term_v1> &terms,
                          std::uint64_t edge_ordinal) {
        const relation_storage *relation = &support_fixture.base;
        std::uint64_t local = edge_ordinal;
        if (local >= relation->edges.size()) {
            local -= relation->edges.size();
            relation = &support_fixture.delta;
        }
        const auto &edge = relation->edges[local];
        for (std::uint32_t k = 0; k < cfg.query_terms; ++k) {
            const auto &p = relation->participants[edge.participant_begin + k];
            terms.push_back({p.role, 0u, p.entity_id});
        }
    };
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        if (cfg.all_terms) {
            append_all(result.primary, splitmix64(rng) % cfg.edges);
            append_all(result.derivative, splitmix64(rng) % cfg.edges);
        } else {
            append_any(result.primary, cell, 0u);
            append_any(result.derivative, cell, 1u);
        }
    }
    for (const auto &term : result.primary) {
        result.hash = hash_mix(result.hash, term.role);
        result.hash = hash_mix(result.hash, term.entity_id);
    }
    for (const auto &term : result.derivative) {
        result.hash = hash_mix(result.hash, term.role);
        result.hash = hash_mix(result.hash, term.entity_id);
    }
    return result;
}

struct candidate_buffer {
    std::vector<std::uint64_t> offsets;
    std::vector<indexing::indexed_work_item_v1> items;
    std::uint64_t written = 0;

    candidate_buffer(std::uint64_t cells, std::uint64_t edge_count)
        : offsets(cells + 1u, 0u), items(cells * edge_count) {}

    gating::per_cell_candidate_view_v1 view() const {
        return {offsets.data(), items.data(), written};
    }
};

bool edge_matches(const support::logical_support_view_v1 &view,
                  const support::logical_support_edge_v1 &edge,
                  const indexing::posting_term_v1 *terms,
                  std::uint64_t term_count, bool all_terms) {
    std::array<std::uint8_t, 3> found{};
    for (std::uint32_t p = 0; p < edge.participant_count; ++p) {
        const auto &participant = view.participants[edge.participant_begin + p];
        const indexing::posting_term_v1 needle{participant.role, 0u,
                                               participant.entity_id};
        const auto *match = std::lower_bound(terms, terms + term_count,
                                             needle, term_less);
        if (match != terms + term_count && match->role == needle.role
            && match->entity_id == needle.entity_id)
            found[static_cast<std::size_t>(match - terms)] = 1u;
    }
    if (all_terms)
        return std::all_of(found.begin(), found.begin() + term_count,
                           [](std::uint8_t value) { return value != 0u; });
    return std::any_of(found.begin(), found.begin() + term_count,
                       [](std::uint8_t value) { return value != 0u; });
}

bool edge_matches(const fixture &f, std::uint64_t ordinal,
                  const indexing::posting_term_v1 *terms,
                  std::uint64_t term_count, bool all_terms) {
    const relation_storage *relation = &f.base;
    std::uint64_t local = ordinal;
    if (local >= relation->edges.size()) {
        local -= relation->edges.size();
        relation = &f.delta;
    }
    return edge_matches(relation->view, relation->edges[local], terms,
                        term_count, all_terms);
}

void scan_batch(const fixture &f, const options &cfg,
                const std::vector<indexing::posting_term_v1> &terms,
                candidate_buffer &out) {
    out.written = 0u;
    out.offsets[0] = 0u;
    const auto mode = cfg.all_terms;
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        const auto *cell_terms = terms.data() + cell * cfg.query_terms;
        for (std::uint64_t ordinal = 0; ordinal < cfg.edges; ++ordinal) {
            if (!edge_matches(f, ordinal, cell_terms, cfg.query_terms, mode)) continue;
            const std::uint64_t id = ordinal + 1u;
            out.items[out.written++] = {id, ordinal};
        }
        out.offsets[cell + 1u] = out.written;
    }
}

void postings_batch(const indexing::reverse_posting_view_v1 &index,
                    const options &cfg,
                    const std::vector<indexing::posting_term_v1> &terms,
                    std::vector<indexing::indexed_work_item_v1> &scratch,
                    candidate_buffer &out) {
    out.written = 0u;
    out.offsets[0] = 0u;
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        const auto *cell_terms = terms.data() + cell * cfg.query_terms;
        auto *dest = out.items.data() + out.written;
        indexing::candidate_output_v1 candidate{dest, cfg.edges, 0u, 0u};
        const auto status = indexing::query_reverse_postings_v1(index,
            cell_terms, cfg.query_terms,
            cfg.all_terms ? indexing::posting_match_v1::all
                          : indexing::posting_match_v1::any,
            {scratch.data(), scratch.size()}, &candidate);
        if (status != indexing::posting_status_v1::success)
            throw std::runtime_error("reverse posting query failed, status="
                + std::to_string(static_cast<unsigned>(status)));
        out.written += candidate.written;
        out.offsets[cell + 1u] = out.written;
    }
}

std::array<gating::condition_node_v1, 7> make_condition_nodes() {
    using gating::condition_node_v1;
    using gating::condition_opcode_v1;
    return {{
        {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 0u, 0u},
        {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 1u, 0u},
        {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 2u, 0u},
        {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 3u, 0u},
        {condition_opcode_v1::all_of, 1u, 0u, 0u, 2u, 0u, 0u},
        {condition_opcode_v1::any_of, 1u, 0u, 2u, 2u, 0u, 0u},
        {condition_opcode_v1::all_of, 1u, 0u, 0u, 3u, 0u, 0u},
    }};
}

struct gate_fixture {
    std::array<gating::condition_node_v1, 7> nodes = make_condition_nodes();
    std::vector<gating::edge_condition_binding_v1> bindings;
    gating::condition_program_view_v1 program{};
    std::vector<std::uint8_t> predicates;
    std::vector<std::uint8_t> node_scratch;
    std::vector<std::uint64_t> bitmaps;
    std::vector<std::uint64_t> dirty_ordinals;
    std::vector<std::uint8_t> changed_predicates;
    std::uint64_t words_per_cell = 0u;
    double actual_gate_density = 0.0;
};

bool root_depends_on_p0(std::uint32_t root) {
    return root == 0u || root == 4u || root == 6u;
}

gate_fixture make_gate_fixture(const options &cfg, const fixture &f) {
    gate_fixture result;
    result.bindings.reserve(cfg.edges);
    const std::uint64_t p0_count = cfg.dirty_fraction == 0.0 ? 0u
        : std::max<std::uint64_t>(1u, static_cast<std::uint64_t>(
            std::ceil(cfg.edges * cfg.dirty_fraction)));
    for (std::uint64_t edge = 0; edge < cfg.edges; ++edge) {
        std::uint32_t root = 0u;
        if (edge < p0_count) root = static_cast<std::uint32_t>(edge % 3u == 0u
            ? 0u : edge % 3u == 1u ? 4u : 6u);
        else {
            const std::array<std::uint32_t, 4> roots{{1u, 2u, 3u, 5u}};
            root = roots[static_cast<std::size_t>((edge - p0_count) % roots.size())];
        }
        result.bindings.push_back({edge, root, 0u});
    }
    result.program = {1u, result.nodes.data(), result.nodes.size(),
        result.bindings.data(), result.bindings.size(), cfg.edges, 4u, 0u};
    const auto predicate_count = cfg.cells * result.program.predicate_count;
    result.predicates.resize(predicate_count);
    std::uint64_t rng = cfg.seed ^ 0x94d049bb133111ebull;
    std::uint64_t passing = 0u;
    for (std::uint64_t i = 0; i < predicate_count; ++i) {
        const double sample = static_cast<double>(splitmix64(rng) >> 11u)
            / static_cast<double>(std::uint64_t{1} << 53u);
        result.predicates[i] = sample < cfg.gate_density ? 1u : 0u;
    }
    result.words_per_cell = cfg.edges / 64u + (cfg.edges % 64u != 0u ? 1u : 0u);
    result.bitmaps.resize(cfg.cells * result.words_per_cell);
    result.node_scratch.resize(result.nodes.size());
    result.changed_predicates = result.predicates;
    if (cfg.cells != 0u) result.changed_predicates[0] ^= 1u;
    for (const auto &binding : result.bindings)
        if (root_depends_on_p0(binding.root_node))
            result.dirty_ordinals.push_back(binding.logical_edge_ordinal);
    if (cfg.dirty_fraction == 0.0) result.dirty_ordinals.clear();
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        for (std::uint64_t edge = 0; edge < cfg.edges; ++edge) {
            const auto root = result.bindings[edge].root_node;
            const auto *p = result.predicates.data() + cell * 4u;
            const bool pass = root == 0u ? p[0] != 0u
                : root == 1u ? p[1] != 0u
                : root == 2u ? p[2] != 0u
                : root == 3u ? p[3] != 0u
                : root == 4u ? p[0] != 0u && p[1] != 0u
                : root == 5u ? p[2] != 0u || p[3] != 0u
                : p[0] != 0u && p[1] != 0u && p[2] != 0u;
            passing += pass;
        }
    }
    result.actual_gate_density = static_cast<double>(passing)
        / static_cast<double>(cfg.cells * cfg.edges);
    (void)f;
    return result;
}

bool condition_pass(const gate_fixture &gate, std::uint64_t cell,
                    std::uint64_t ordinal) {
    const auto root = gate.bindings[ordinal].root_node;
    const auto *p = gate.predicates.data() + cell * 4u;
    switch (root) {
    case 0u: return p[0] != 0u;
    case 1u: return p[1] != 0u;
    case 2u: return p[2] != 0u;
    case 3u: return p[3] != 0u;
    case 4u: return p[0] != 0u && p[1] != 0u;
    case 5u: return p[2] != 0u || p[3] != 0u;
    default: return p[0] != 0u && p[1] != 0u && p[2] != 0u;
    }
}

bool bitmap_test(const std::uint64_t *words, std::uint64_t ordinal) {
    return ((words[ordinal / 64u] >> (ordinal % 64u)) & 1u) != 0u;
}

struct workset_buffers {
    std::vector<std::uint64_t> primal_offsets;
    std::vector<std::uint64_t> derivative_offsets;
    std::vector<indexing::indexed_work_item_v1> primal_items;
    std::vector<indexing::indexed_work_item_v1> derivative_items;
    gating::gate_workset_pair_v1 pair{};

    workset_buffers(std::uint64_t cells, std::uint64_t edge_count)
        : primal_offsets(cells + 1u), derivative_offsets(cells + 1u),
          primal_items(cells * edge_count), derivative_items(cells * edge_count) {
        pair.primal = {primal_offsets.data(), primal_offsets.size(),
                       primal_items.data(), primal_items.size(), 0u, 0u};
        pair.derivative = {derivative_offsets.data(), derivative_offsets.size(),
                           derivative_items.data(), derivative_items.size(), 0u, 0u};
    }
};

void build_worksets(const gate_fixture &gate, const options &cfg,
                    const candidate_buffer &primal,
                    const candidate_buffer &derivative,
                    workset_buffers &out) {
    const auto status = gating::build_condition_worksets_v1(gate.program,
        cfg.cells, 1u, gate.bitmaps.data(), gate.bitmaps.size(),
        primal.view(), derivative.view(), &out.pair);
    if (status != gating::condition_status_v1::success)
        throw std::runtime_error("condition workset build failed, status="
            + std::to_string(static_cast<unsigned>(status)));
}

void validate_candidates_equal(const candidate_buffer &a,
                               const candidate_buffer &b,
                               std::uint64_t cells) {
    if (a.written != b.written || a.offsets != b.offsets)
        throw std::runtime_error("rescan/postings candidate counts differ");
    for (std::uint64_t i = 0; i < a.written; ++i)
        if (a.items[i].logical_edge_id != b.items[i].logical_edge_id
            || a.items[i].logical_edge_ordinal != b.items[i].logical_edge_ordinal)
            throw std::runtime_error("rescan/postings candidate IDs differ");
    (void)cells;
}

void validate_gate_oracle(const gate_fixture &gate, const options &cfg,
                          const candidate_buffer &primal,
                          const candidate_buffer &derivative,
                          const workset_buffers &out,
                          std::uint64_t *derivative_only) {
    if (out.pair.primal.structure_epoch != gate.program.structure_epoch
        || out.pair.derivative.structure_epoch != gate.program.structure_epoch
        || out.pair.primal.state_generation != 1u
        || out.pair.derivative.state_generation != 1u)
        throw std::runtime_error("condition workset epoch metadata mismatch");
    std::uint64_t derivative_only_count = 0u;
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        std::vector<indexing::indexed_work_item_v1> expected_primal;
        std::vector<indexing::indexed_work_item_v1> expected_derivative;
        for (std::uint64_t i = primal.offsets[cell]; i < primal.offsets[cell + 1u]; ++i)
            if (condition_pass(gate, cell, primal.items[i].logical_edge_ordinal))
                expected_primal.push_back(primal.items[i]);
        for (std::uint64_t i = primal.offsets[cell]; i < primal.offsets[cell + 1u]; ++i)
            if (condition_pass(gate, cell, primal.items[i].logical_edge_ordinal))
                expected_derivative.push_back(primal.items[i]);
        for (std::uint64_t i = derivative.offsets[cell];
             i < derivative.offsets[cell + 1u]; ++i)
            if (condition_pass(gate, cell, derivative.items[i].logical_edge_ordinal))
                expected_derivative.push_back(derivative.items[i]);
        const auto by_id = [](const auto &x, const auto &y) {
            return x.logical_edge_id < y.logical_edge_id;
        };
        const auto unique = [](auto &items) {
            std::sort(items.begin(), items.end(), [](const auto &x, const auto &y) {
                return x.logical_edge_id < y.logical_edge_id;
            });
            items.erase(std::unique(items.begin(), items.end(), [](const auto &x,
                const auto &y) { return x.logical_edge_id == y.logical_edge_id; }),
                items.end());
        };
        unique(expected_primal);
        unique(expected_derivative);
        const auto p_begin = out.pair.primal.offsets[cell];
        const auto p_end = out.pair.primal.offsets[cell + 1u];
        const auto d_begin = out.pair.derivative.offsets[cell];
        const auto d_end = out.pair.derivative.offsets[cell + 1u];
        if (p_end - p_begin != expected_primal.size()
            || d_end - d_begin != expected_derivative.size())
            throw std::runtime_error("condition workset oracle count mismatch");
        for (std::size_t i = 0; i < expected_primal.size(); ++i)
            if (out.pair.primal.items[p_begin + i].logical_edge_id
                    != expected_primal[i].logical_edge_id
                || out.pair.primal.items[p_begin + i].logical_edge_ordinal
                    != expected_primal[i].logical_edge_ordinal)
                throw std::runtime_error("condition primal workset oracle mismatch");
        for (std::size_t i = 0; i < expected_derivative.size(); ++i)
            if (out.pair.derivative.items[d_begin + i].logical_edge_id
                    != expected_derivative[i].logical_edge_id
                || out.pair.derivative.items[d_begin + i].logical_edge_ordinal
                    != expected_derivative[i].logical_edge_ordinal)
                throw std::runtime_error("condition derivative workset oracle mismatch");
        for (const auto &item : expected_derivative)
            if (!std::binary_search(expected_primal.begin(), expected_primal.end(),
                                    item, by_id))
                ++derivative_only_count;
    }
    if (derivative_only) *derivative_only = derivative_only_count;
}

struct timing_stats {
    std::vector<double> samples;
    double minimum = 0.0;
    double p10 = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    double maximum = 0.0;
};

double percentile(const std::vector<double> &sorted, double p) {
    const double position = p * static_cast<double>(sorted.size() - 1u);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1u, sorted.size() - 1u);
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

template <class Function>
timing_stats measure(std::uint32_t sample_count, std::uint64_t iterations,
                     Function &&function) {
    volatile std::uint64_t sink = 0u;
    for (unsigned warm = 0; warm < 3; ++warm) sink += function();
    timing_stats result;
    result.samples.reserve(sample_count);
    for (std::uint32_t i = 0; i < sample_count; ++i) {
        const auto begin = clock_type::now();
        for (std::uint64_t iter = 0; iter < iterations; ++iter)
            sink += function();
        const auto end = clock_type::now();
        const double elapsed = std::chrono::duration<double, std::micro>(end - begin).count();
        result.samples.push_back(elapsed / static_cast<double>(iterations));
    }
    (void)sink;
    auto sorted = result.samples;
    std::sort(sorted.begin(), sorted.end());
    result.minimum = sorted.front();
    result.p10 = percentile(sorted, 0.10);
    result.median = percentile(sorted, 0.50);
    result.p90 = percentile(sorted, 0.90);
    result.maximum = sorted.back();
    return result;
}

void print_timing(const char *name, const timing_stats &stats, bool comma) {
    std::cout << "\"" << name << "\":{\"min_us\":" << stats.minimum
        << ",\"p10_us\":" << stats.p10 << ",\"median_us\":" << stats.median
        << ",\"p90_us\":" << stats.p90 << ",\"max_us\":" << stats.maximum
        << ",\"samples_us\":[";
    for (std::size_t i = 0; i < stats.samples.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << stats.samples[i];
    }
    std::cout << "]}" << (comma ? "," : "");
}

bool same_bitmap(const std::uint64_t *a, const std::uint64_t *b,
                 std::uint64_t words) {
    return std::equal(a, a + words, b);
}

void validate_bitmap_oracle(const gate_fixture &gate, const options &cfg) {
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell)
        for (std::uint64_t edge = 0; edge < cfg.edges; ++edge)
            if (bitmap_test(gate.bitmaps.data() + cell * gate.words_per_cell, edge)
                != condition_pass(gate, cell, edge))
                throw std::runtime_error("condition bitmap oracle mismatch");
}

void print_hex(std::uint64_t value) {
    std::cout << "\"0x" << std::hex << value << std::dec << "\"";
}

void run(const options &cfg) {
    fixture f = make_fixture(cfg);
    const auto terms = make_terms(cfg, f);
    const auto requirements = indexing::query_reverse_posting_requirements_v1(f.snapshot);
    std::vector<indexing::reverse_posting_record_v1> record_scratch(
        requirements.record_capacity);
    std::vector<indexing::reverse_posting_record_v1> posting_entries(
        requirements.maximum_unique_entries);
    indexing::reverse_posting_view_v1 index{};
    auto build_index = [&]() -> std::uint64_t {
        const auto status = indexing::build_reverse_postings_v1(f.snapshot,
            record_scratch.data(), record_scratch.size(), posting_entries.data(),
            posting_entries.size(), &index);
        if (status != indexing::posting_status_v1::success)
            throw std::runtime_error("reverse posting build failed, status="
                + std::to_string(static_cast<unsigned>(status)));
        return index.entry_count;
    };
    const auto build_stats = measure(cfg.samples, 1u, build_index);
    (void)build_index();

    candidate_buffer scan_primal(cfg.cells, cfg.edges);
    candidate_buffer scan_derivative(cfg.cells, cfg.edges);
    candidate_buffer post_primal(cfg.cells, cfg.edges);
    candidate_buffer post_derivative(cfg.cells, cfg.edges);
    std::vector<indexing::indexed_work_item_v1> query_scratch(
        std::max<std::uint64_t>(1u, index.entry_count * cfg.query_terms));
    auto do_scan_pair = [&]() -> std::uint64_t {
        scan_batch(f, cfg, terms.primary, scan_primal);
        scan_batch(f, cfg, terms.derivative, scan_derivative);
        return scan_primal.written + scan_derivative.written;
    };
    auto do_post_pair = [&]() -> std::uint64_t {
        postings_batch(index, cfg, terms.primary, query_scratch, post_primal);
        postings_batch(index, cfg, terms.derivative, query_scratch, post_derivative);
        return post_primal.written + post_derivative.written;
    };
    const std::uint64_t visits = cfg.edges * cfg.cells;
    const std::uint64_t iterations = std::max<std::uint64_t>(1u,
        std::min<std::uint64_t>(64u, cfg.target_edge_visits / std::max<std::uint64_t>(1u, visits)));
    const auto scan_stats = measure(cfg.samples, iterations, do_scan_pair);
    const auto post_stats = measure(cfg.samples, iterations, do_post_pair);
    (void)do_scan_pair();
    (void)do_post_pair();
    validate_candidates_equal(scan_primal, post_primal, cfg.cells);
    validate_candidates_equal(scan_derivative, post_derivative, cfg.cells);

    gate_fixture gate = make_gate_fixture(cfg, f);
    const auto bitmap_status = gating::evaluate_condition_bitmaps_v1(gate.program,
        cfg.cells, gate.predicates.data(), gate.predicates.size(), gate.bitmaps.data(),
        gate.bitmaps.size(), gate.node_scratch.data(), gate.node_scratch.size());
    if (bitmap_status != gating::condition_status_v1::success)
        throw std::runtime_error("condition bitmap build failed, status="
            + std::to_string(static_cast<unsigned>(bitmap_status)));
    validate_bitmap_oracle(gate, cfg);
    for (const auto &binding : gate.bindings) {
        f.hash = hash_mix(f.hash, binding.logical_edge_ordinal);
        f.hash = hash_mix(f.hash, binding.root_node);
    }
    for (auto value : gate.predicates) f.hash = hash_mix(f.hash, value);
    for (auto value : terms.primary) {
        f.hash = hash_mix(f.hash, value.role);
        f.hash = hash_mix(f.hash, value.entity_id);
    }
    const auto bitmap_stats = measure(cfg.samples, iterations, [&]() -> std::uint64_t {
        const auto status = gating::evaluate_condition_bitmaps_v1(gate.program,
            cfg.cells, gate.predicates.data(), gate.predicates.size(), gate.bitmaps.data(),
            gate.bitmaps.size(), gate.node_scratch.data(), gate.node_scratch.size());
        if (status != gating::condition_status_v1::success)
            throw std::runtime_error("timed bitmap evaluation failed");
        return gate.bitmaps.size();
    });

    auto changed_bitmap = std::vector<std::uint64_t>(
        gate.bitmaps.begin(), gate.bitmaps.begin() + gate.words_per_cell);
    auto expected_changed_bitmap = changed_bitmap;
    const auto dirty_status = gating::update_condition_bitmap_v1(gate.program,
        gate.changed_predicates.data(), gate.program.predicate_count,
        gate.dirty_ordinals.data(), gate.dirty_ordinals.size(),
        changed_bitmap.data(), changed_bitmap.size(), gate.node_scratch.data(),
        gate.node_scratch.size());
    if (dirty_status != gating::condition_status_v1::success)
        throw std::runtime_error("dirty bitmap update failed");
    const auto full_changed_status = gating::evaluate_condition_bitmaps_v1(
        gate.program, 1u, gate.changed_predicates.data(), gate.program.predicate_count,
        expected_changed_bitmap.data(), expected_changed_bitmap.size(),
        gate.node_scratch.data(), gate.node_scratch.size());
    if (full_changed_status != gating::condition_status_v1::success
        || !same_bitmap(changed_bitmap.data(), expected_changed_bitmap.data(),
                        gate.words_per_cell))
        throw std::runtime_error("dirty bitmap update differs from full recompute");
    const auto full_cell_bitmap_stats = measure(cfg.samples, iterations,
        [&]() -> std::uint64_t {
            const auto status = gating::evaluate_condition_bitmaps_v1(gate.program,
                1u, gate.changed_predicates.data(), gate.program.predicate_count,
                expected_changed_bitmap.data(), expected_changed_bitmap.size(),
                gate.node_scratch.data(), gate.node_scratch.size());
            if (status != gating::condition_status_v1::success)
                throw std::runtime_error("timed one-cell bitmap evaluation failed");
            return gate.words_per_cell;
        });
    const auto dirty_stats = measure(cfg.samples, iterations, [&]() -> std::uint64_t {
        const auto status = gating::update_condition_bitmap_v1(gate.program,
            gate.changed_predicates.data(), gate.program.predicate_count,
            gate.dirty_ordinals.data(), gate.dirty_ordinals.size(),
            changed_bitmap.data(), changed_bitmap.size(), gate.node_scratch.data(),
            gate.node_scratch.size());
        if (status != gating::condition_status_v1::success)
            throw std::runtime_error("timed dirty bitmap update failed");
        return gate.dirty_ordinals.size();
    });

    workset_buffers work_scan(cfg.cells, cfg.edges);
    workset_buffers work_post(cfg.cells, cfg.edges);
    build_worksets(gate, cfg, scan_primal, scan_derivative, work_scan);
    build_worksets(gate, cfg, post_primal, post_derivative, work_post);
    std::uint64_t derivative_only = 0u;
    validate_gate_oracle(gate, cfg, scan_primal, scan_derivative,
                         work_scan, &derivative_only);
    validate_gate_oracle(gate, cfg, post_primal, post_derivative,
                         work_post, nullptr);
    if (work_scan.pair.primal.written != work_post.pair.primal.written
        || work_scan.pair.derivative.written != work_post.pair.derivative.written)
        throw std::runtime_error("postings and scan gate workset counts differ");
    for (std::uint64_t i = 0; i < work_scan.pair.primal.written; ++i)
        if (work_scan.pair.primal.items[i].logical_edge_id
                != work_post.pair.primal.items[i].logical_edge_id
            || work_scan.pair.primal.items[i].logical_edge_ordinal
                != work_post.pair.primal.items[i].logical_edge_ordinal)
            throw std::runtime_error("postings and scan primal worksets differ");
    for (std::uint64_t i = 0; i < work_scan.pair.derivative.written; ++i)
        if (work_scan.pair.derivative.items[i].logical_edge_id
                != work_post.pair.derivative.items[i].logical_edge_id
            || work_scan.pair.derivative.items[i].logical_edge_ordinal
                != work_post.pair.derivative.items[i].logical_edge_ordinal)
            throw std::runtime_error("postings and scan derivative worksets differ");

    const std::uint64_t pair_candidate_count = scan_primal.written + scan_derivative.written;
    std::uint64_t max_primal_per_cell = 0u;
    std::uint64_t max_derivative_per_cell = 0u;
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        max_primal_per_cell = std::max(max_primal_per_cell,
            scan_primal.offsets[cell + 1u] - scan_primal.offsets[cell]);
        max_derivative_per_cell = std::max(max_derivative_per_cell,
            scan_derivative.offsets[cell + 1u] - scan_derivative.offsets[cell]);
    }
    const std::uint64_t workset_input_items = pair_candidate_count;
    const bool exceeds_workset_volume_cap = workset_input_items > 4000000u;
    const bool skip_workset_timing = cfg.skip_worksets
        || exceeds_workset_volume_cap;
    timing_stats workset_scan_stats{};
    timing_stats workset_post_stats{};
    timing_stats pipeline_scan_stats{};
    timing_stats pipeline_post_stats{};
    if (!skip_workset_timing) {
        workset_scan_stats = measure(cfg.samples, iterations, [&]() -> std::uint64_t {
            build_worksets(gate, cfg, scan_primal, scan_derivative, work_scan);
            return work_scan.pair.derivative.written;
        });
        workset_post_stats = measure(cfg.samples, iterations, [&]() -> std::uint64_t {
            build_worksets(gate, cfg, post_primal, post_derivative, work_post);
            return work_post.pair.derivative.written;
        });
        pipeline_scan_stats = measure(cfg.samples, iterations, [&]() -> std::uint64_t {
            scan_batch(f, cfg, terms.primary, scan_primal);
            scan_batch(f, cfg, terms.derivative, scan_derivative);
            const auto status = gating::evaluate_condition_bitmaps_v1(gate.program,
                cfg.cells, gate.predicates.data(), gate.predicates.size(),
                gate.bitmaps.data(), gate.bitmaps.size(), gate.node_scratch.data(),
                gate.node_scratch.size());
            if (status != gating::condition_status_v1::success)
                throw std::runtime_error("scan pipeline bitmap failure");
            build_worksets(gate, cfg, scan_primal, scan_derivative, work_scan);
            return work_scan.pair.derivative.written;
        });
        pipeline_post_stats = measure(cfg.samples, iterations, [&]() -> std::uint64_t {
            postings_batch(index, cfg, terms.primary, query_scratch, post_primal);
            postings_batch(index, cfg, terms.derivative, query_scratch, post_derivative);
            const auto status = gating::evaluate_condition_bitmaps_v1(gate.program,
                cfg.cells, gate.predicates.data(), gate.predicates.size(),
                gate.bitmaps.data(), gate.bitmaps.size(), gate.node_scratch.data(),
                gate.node_scratch.size());
            if (status != gating::condition_status_v1::success)
                throw std::runtime_error("postings pipeline bitmap failure");
            build_worksets(gate, cfg, post_primal, post_derivative, work_post);
            return work_post.pair.derivative.written;
        });
    }

    std::cout << std::setprecision(9)
        << "{\"schema\":\"cellerator.sparse_support.host_index_bench.v1\","
        << "\"scope\":\"host_only_no_cuda\",\"correctness\":\"exact_oracle_pass\","
        << "\"fixture\":{\"edges\":" << cfg.edges
        << ",\"cells\":" << cfg.cells << ",\"entities\":" << cfg.entities
        << ",\"participants_per_edge\":3,\"delta_edges\":"
        << f.delta.edges.size() << ",\"query_terms\":" << cfg.query_terms
        << ",\"match\":\"" << (cfg.all_terms ? "all" : "any")
        << "\",\"gate_predicate_density\":" << cfg.gate_density
        << ",\"actual_gate_density\":" << gate.actual_gate_density
        << ",\"source_skew\":" << cfg.skew
        << ",\"p0_dirty_edge_fraction\":" << cfg.dirty_fraction
        << ",\"seed\":" << cfg.seed << ",\"digest\":";
    print_hex(hash_mix(f.hash, terms.hash));
    std::cout << "},\"counts\":{\"postings_entries\":" << index.entry_count
        << ",\"primal_candidates\":" << scan_primal.written
        << ",\"derivative_candidates\":" << scan_derivative.written
        << ",\"primal_work_items\":" << work_scan.pair.primal.written
        << ",\"derivative_union_work_items\":" << work_scan.pair.derivative.written
        << ",\"derivative_only_work_items\":" << derivative_only
        << ",\"dirty_edges\":" << gate.dirty_ordinals.size()
        << ",\"max_primal_candidates_per_cell\":" << max_primal_per_cell
        << ",\"max_derivative_candidates_per_cell\":" << max_derivative_per_cell
        << ",\"query_iterations_per_sample\":" << iterations << "},"
        << "\"memory_bytes\":{\"support_base_delta\":"
        << (f.base.edges.capacity() * sizeof(support::logical_support_edge_v1)
            + f.base.participants.capacity() * sizeof(support::logical_support_participant_v1)
            + f.delta.edges.capacity() * sizeof(support::logical_support_edge_v1)
            + f.delta.participants.capacity() * sizeof(support::logical_support_participant_v1))
        << ",\"posting_index\":" << posting_entries.capacity()
            * sizeof(indexing::reverse_posting_record_v1)
        << ",\"posting_build_scratch\":" << record_scratch.capacity()
            * sizeof(indexing::reverse_posting_record_v1)
        << ",\"query_scratch\":" << query_scratch.capacity()
            * sizeof(indexing::indexed_work_item_v1)
        << ",\"query_terms\":" << (terms.primary.capacity()
                + terms.derivative.capacity()) * sizeof(indexing::posting_term_v1)
        << ",\"condition_program\":"
            << gate.bindings.capacity() * sizeof(gating::edge_condition_binding_v1)
                + gate.nodes.size() * sizeof(gating::condition_node_v1)
        << ",\"condition_bitmaps\":" << gate.bitmaps.capacity() * sizeof(std::uint64_t)
        << ",\"condition_state_and_temporaries\":"
            << (gate.predicates.capacity() + gate.node_scratch.capacity()
                + gate.changed_predicates.capacity()) * sizeof(std::uint8_t)
                + (gate.dirty_ordinals.capacity() + gate.words_per_cell * 2u)
                    * sizeof(std::uint64_t)
        << ",\"candidate_buffers\":"
            << (scan_primal.items.capacity() + scan_derivative.items.capacity()
                + post_primal.items.capacity() + post_derivative.items.capacity())
                * sizeof(indexing::indexed_work_item_v1)
            + (scan_primal.offsets.capacity() + scan_derivative.offsets.capacity()
                + post_primal.offsets.capacity() + post_derivative.offsets.capacity())
                * sizeof(std::uint64_t)
        << ",\"gate_workset_buffers\":"
            << (work_scan.primal_items.capacity() + work_scan.derivative_items.capacity()
                + work_post.primal_items.capacity() + work_post.derivative_items.capacity())
                * sizeof(indexing::indexed_work_item_v1)
            + (work_scan.primal_offsets.capacity() + work_scan.derivative_offsets.capacity()
                + work_post.primal_offsets.capacity() + work_post.derivative_offsets.capacity())
                * sizeof(std::uint64_t)
        << "},\"timings_us\":{";
    print_timing("posting_index_build", build_stats, true);
    print_timing("full_rescan_pair_discovery", scan_stats, true);
    print_timing("reverse_posting_pair_discovery", post_stats, true);
    print_timing("condition_bitmap_full_rebuild", bitmap_stats, true);
    print_timing("condition_bitmap_full_rebuild_one_cell", full_cell_bitmap_stats, true);
    print_timing("condition_bitmap_dirty_update", dirty_stats, true);
    if (!skip_workset_timing) {
        print_timing("gate_worksets_after_rescan", workset_scan_stats, true);
        print_timing("gate_worksets_after_postings", workset_post_stats, true);
        print_timing("complete_rescan_bitmap_workset", pipeline_scan_stats, true);
        print_timing("complete_postings_bitmap_workset", pipeline_post_stats, false);
    } else {
        std::cout << "\"gate_worksets_after_rescan\":null,"
                     "\"gate_worksets_after_postings\":null,"
                     "\"complete_rescan_bitmap_workset\":null,"
                     "\"complete_postings_bitmap_workset\":null";
    }
    std::cout << "},\"workset_timing_status\":\""
        << (cfg.skip_worksets ? "skipped_by_request"
            : exceeds_workset_volume_cap ? "skipped_candidate_volume_cap" : "measured")
        << "\",\"workset_input_candidate_items\":" << workset_input_items
        << ",\"reuse_amortization_us\":{\"posting_setup_plus_pipeline_reuse_1\":";
    if (!skip_workset_timing)
        std::cout << build_stats.median + pipeline_post_stats.median;
    else std::cout << "null";
    std::cout << ",\"posting_setup_plus_pipeline_reuse_16\":";
    if (!skip_workset_timing)
        std::cout << build_stats.median / 16.0 + pipeline_post_stats.median;
    else std::cout << "null";
    std::cout << "},\"host\":{\"hardware_threads\":"
        << std::thread::hardware_concurrency() << ",\"compiler\":\"" << __VERSION__
        << "\",\"samples\":" << cfg.samples
        << ",\"warmups_per_measurement\":3,\"target_edge_visits_per_sample\":"
        << cfg.target_edge_visits << "}}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto cfg = parse_options(argc, argv);
        run(cfg);
    } catch (const std::exception &error) {
        std::cerr << "sparse_support_index_bench: " << error.what() << '\n';
        return 2;
    }
    return 0;
}
