#include <Cellerator/compute/gating/condition_program.hh>
#include <Cellerator/compute/gating/cuda_worksets.cuh>
#include <Cellerator/compute/indexing/cuda_postings.cuh>
#include <Cellerator/compute/support/logical_support.hh>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <functional>
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
using host_clock = std::chrono::steady_clock;

#define CUDA_CHECK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) throw std::runtime_error( \
        std::string(#call) + ": " + cudaGetErrorString(error)); \
} while (0)

struct options {
    std::string case_id = "anchor_e2176_b64";
    std::uint64_t edges = 2176;
    std::uint64_t cells = 64;
    std::uint64_t entities = 256;
    double gate_density = 0.5;
    double source_skew = 0.0;
    double dirty_fraction = 0.01;
    std::uint64_t seed = 20261027;
    std::uint32_t warmup = 3;
    std::uint32_t samples = 21;
    std::uint32_t iterations = 5;
};

std::uint64_t random_next(std::uint64_t &state) {
    state += 0x9e3779b97f4a7c15ull;
    std::uint64_t x = state;
    x = (x ^ (x >> 30u)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27u)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31u);
}

std::uint64_t mix_hash(std::uint64_t h, std::uint64_t value) {
    value ^= value >> 30u;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27u;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31u;
    return h ^ (value + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u));
}

double unit_random(std::uint64_t &state) {
    return static_cast<double>(random_next(state) >> 11u)
        / static_cast<double>(std::uint64_t{1} << 53u);
}

std::uint64_t parse_u64(const char *value, const char *name) {
    char *end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0')
        throw std::runtime_error(std::string("invalid ") + name);
    return static_cast<std::uint64_t>(parsed);
}

double parse_double(const char *value, const char *name) {
    char *end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || *end != '\0' || !std::isfinite(parsed))
        throw std::runtime_error(std::string("invalid ") + name);
    return parsed;
}

options parse_options(int argc, char **argv) {
    options result;
    auto value = [&](int &i, const char *name) {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing ") + name);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case-id") result.case_id = value(i, "--case-id");
        else if (arg == "--edges") result.edges = parse_u64(value(i, "--edges"), "edges");
        else if (arg == "--cells") result.cells = parse_u64(value(i, "--cells"), "cells");
        else if (arg == "--entities") result.entities = parse_u64(value(i, "--entities"), "entities");
        else if (arg == "--gate-density") result.gate_density = parse_double(value(i, "--gate-density"), "gate density");
        else if (arg == "--source-skew") result.source_skew = parse_double(value(i, "--source-skew"), "source skew");
        else if (arg == "--dirty-fraction") result.dirty_fraction = parse_double(value(i, "--dirty-fraction"), "dirty fraction");
        else if (arg == "--seed") result.seed = parse_u64(value(i, "--seed"), "seed");
        else if (arg == "--warmup") result.warmup = static_cast<std::uint32_t>(parse_u64(value(i, "--warmup"), "warmup"));
        else if (arg == "--samples") result.samples = static_cast<std::uint32_t>(parse_u64(value(i, "--samples"), "samples"));
        else if (arg == "--iterations") result.iterations = static_cast<std::uint32_t>(parse_u64(value(i, "--iterations"), "iterations"));
        else if (arg == "--help") {
            std::cout << "cuda_indexing_bench --case-id ID --edges N --cells N "
                         "--entities N --gate-density P --source-skew P "
                         "--dirty-fraction P --warmup N --samples N --iterations N\n";
            std::exit(0);
        } else throw std::runtime_error("unknown option: " + arg);
    }
    if (result.edges == 0 || result.cells == 0 || result.cells > 4096
        || result.entities < 3 || result.warmup > 100 || result.samples < 3
        || result.iterations == 0 || result.iterations > 1000
        || result.gate_density < 0 || result.gate_density > 1
        || result.source_skew < 0 || result.source_skew > 1
        || result.dirty_fraction < 0 || result.dirty_fraction > 1
        || result.edges > std::numeric_limits<std::uint64_t>::max() / result.cells)
        throw std::runtime_error("unsupported dimensions or probabilities");
    const auto candidate_items = result.edges * result.cells;
    if (candidate_items > static_cast<std::uint64_t>(std::numeric_limits<int>::max())
        || candidate_items > std::numeric_limits<std::uint64_t>::max() / 2u)
        throw std::runtime_error("candidate volume exceeds scan or workspace limits");
    return result;
}

template <class T> struct device_buffer {
    T *data = nullptr;
    std::size_t size = 0;
    static inline std::uint64_t requested_bytes = 0u;
    device_buffer() = default;
    explicit device_buffer(std::size_t count) : size(count) {
        if (size) {
            if (size > std::numeric_limits<std::size_t>::max() / sizeof(T))
                throw std::runtime_error("device allocation size overflow");
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data), size * sizeof(T)));
            requested_bytes += static_cast<std::uint64_t>(size * sizeof(T));
        }
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    ~device_buffer() { if (data) cudaFree(data); }
};

template <class T> void copy_to_device(device_buffer<T> &destination,
                                       const std::vector<T> &source,
                                       cudaStream_t stream) {
    if (destination.size < source.size()) throw std::runtime_error("device copy capacity too small");
    if (!source.empty()) CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data(),
        source.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
}

template <class T> std::vector<T> copy_from_device(const T *source, std::size_t count) {
    std::vector<T> result(count);
    if (count) CUDA_CHECK(cudaMemcpy(result.data(), source, count * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}

std::uint64_t total_requested_device_bytes() {
    return device_buffer<support::logical_support_edge_v1>::requested_bytes
        + device_buffer<support::logical_support_participant_v1>::requested_bytes
        + device_buffer<indexing::posting_term_v1>::requested_bytes
        + device_buffer<std::uint64_t>::requested_bytes
        + device_buffer<indexing::indexed_work_item_v1>::requested_bytes
        + device_buffer<std::uint8_t>::requested_bytes
        + device_buffer<gating::condition_node_v1>::requested_bytes
        + device_buffer<gating::edge_condition_binding_v1>::requested_bytes
        + device_buffer<gating::cuda_workset_status_v1>::requested_bytes
        + device_buffer<indexing::cuda_posting_scratch_v1::cursor_v1>::requested_bytes
        + device_buffer<indexing::cuda_posting_status_v1>::requested_bytes;
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
    std::vector<support::logical_support_edge_v1> gpu_edges;
    std::vector<support::logical_support_participant_v1> gpu_participants;
    std::uint64_t digest = 1469598103934665603ull;
};

std::uint64_t choose_source(std::uint64_t &rng, const options &cfg) {
    if (unit_random(rng) < cfg.source_skew) return 1u;
    return 1u + random_next(rng) % cfg.entities;
}

fixture make_fixture(const options &cfg) {
    fixture f;
    const auto delta_edges = cfg.edges > 1 ? std::max<std::uint64_t>(1u, cfg.edges / 16u) : 0u;
    const auto base_edges = cfg.edges - delta_edges;
    std::uint64_t rng = cfg.seed;
    auto fill = [&](relation_storage &r, std::uint64_t ordinal_begin,
                    std::uint64_t count, std::uint64_t relation_id) {
        r.edges.reserve(count);
        r.participants.reserve(count * 4u);
        for (std::uint64_t local = 0; local < count; ++local) {
            const auto ordinal = ordinal_begin + local;
            const auto source = choose_source(rng, cfg);
            const auto target = 1u + random_next(rng) % cfg.entities;
            const auto cofactor = 1u + random_next(rng) % cfg.entities;
            const auto begin = r.participants.size();
            r.participants.push_back({0u, 0u, source});
            r.participants.push_back({1u, 0u, target});
            r.participants.push_back({2u, 0u, cofactor});
            const std::uint32_t count_in_edge = ordinal % 31u == 0u ? 4u : 3u;
            if (count_in_edge == 4u) r.participants.push_back({0u, 0u, source});
            r.edges.push_back({ordinal + 1u, begin, count_in_edge, 0u});
            f.digest = mix_hash(f.digest, ordinal + 1u);
            f.digest = mix_hash(f.digest, source);
            f.digest = mix_hash(f.digest, target);
            f.digest = mix_hash(f.digest, cofactor);
            f.digest = mix_hash(f.digest, count_in_edge);
        }
        r.view = {support::logical_support_schema_version_v1, 0u,
            relation_id, 9001u, 1u, r.edges.data(), r.edges.size(),
            r.participants.data(), r.participants.size()};
    };
    fill(f.base, 0u, base_edges, 7001u);
    if (delta_edges) fill(f.delta, base_edges, delta_edges, 7002u);
    f.snapshot = {&f.base.view, delta_edges ? &f.delta.view : nullptr, 1u};
    if (support::validate_support_snapshot_pair_v1(f.snapshot)
        != support::support_snapshot_status_v1::success)
        throw std::runtime_error("support fixture validation failed");
    f.gpu_edges.reserve(cfg.edges);
    f.gpu_participants.reserve(f.base.participants.size() + f.delta.participants.size());
    auto append_gpu = [&](const relation_storage &r) {
        const auto participant_base = f.gpu_participants.size();
        f.gpu_participants.insert(f.gpu_participants.end(),
            r.participants.begin(), r.participants.end());
        for (const auto &edge : r.edges)
            f.gpu_edges.push_back({edge.logical_edge_id,
                edge.participant_begin + participant_base,
                edge.participant_count, 0u});
    };
    append_gpu(f.base);
    if (delta_edges) append_gpu(f.delta);
    return f;
}

struct host_candidates {
    std::vector<std::uint64_t> offsets;
    std::vector<indexing::indexed_work_item_v1> items;
    std::uint64_t count = 0;
};

host_candidates make_host_candidates(const options &cfg) {
    return {std::vector<std::uint64_t>(cfg.cells + 1u, 0u),
            std::vector<indexing::indexed_work_item_v1>(cfg.cells * cfg.edges), 0u};
}

struct inputs {
    std::vector<indexing::posting_term_v1> primal_queries;
    std::vector<indexing::posting_term_v1> derivative_queries;
    std::vector<std::uint64_t> query_offsets;
    std::vector<std::uint64_t> dirty_offsets;
    std::vector<std::uint64_t> dirty_edges;
    std::vector<std::uint8_t> predicates;
    std::vector<std::uint8_t> changed_predicates;
    std::array<gating::condition_node_v1, 7> nodes{};
    std::vector<gating::edge_condition_binding_v1> bindings;
    gating::condition_program_view_v1 program{};
    std::uint64_t dirty_per_cell = 0;
    double observed_gate_density = 0;
    std::uint64_t digest = 1469598103934665603ull;
};

inputs make_inputs(const options &cfg, const fixture &f) {
    inputs in;
    in.digest = mix_hash(in.digest, cfg.edges);
    in.digest = mix_hash(in.digest, cfg.cells);
    in.digest = mix_hash(in.digest, cfg.entities);
    in.digest = mix_hash(in.digest, cfg.seed);
    in.primal_queries.reserve(cfg.cells);
    in.derivative_queries.reserve(cfg.cells);
    in.query_offsets.resize(cfg.cells + 1u);
    std::uint64_t rng = cfg.seed ^ 0xd1b54a32d192ed03ull;
    auto choose_query = [&](std::uint32_t role, std::uint64_t cell) {
        const bool hot = role == 0u && cfg.source_skew > 0.5 && cell % 4u == 0u;
        return hot ? 1u : 1u + random_next(rng) % cfg.entities;
    };
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        in.query_offsets[cell] = cell;
        in.primal_queries.push_back({0u, 0u, choose_query(0u, cell)});
        in.derivative_queries.push_back({1u, 0u, choose_query(1u, cell)});
    }
    in.query_offsets[cfg.cells] = cfg.cells;
    using gating::condition_opcode_v1;
    in.nodes = {{
        {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 0u, 0u},
        {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 1u, 0u},
        {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 2u, 0u},
        {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 3u, 0u},
        {condition_opcode_v1::all_of, 1u, 0u, 0u, 2u, 0u, 0u},
        {condition_opcode_v1::any_of, 1u, 0u, 2u, 2u, 0u, 0u},
        {condition_opcode_v1::all_of, 1u, 0u, 0u, 3u, 0u, 0u},
    }};
    in.dirty_per_cell = cfg.dirty_fraction == 0.0 ? 0u
        : std::max<std::uint64_t>(1u, static_cast<std::uint64_t>(
            std::ceil(cfg.edges * cfg.dirty_fraction)));
    in.bindings.reserve(cfg.edges);
    for (std::uint64_t edge = 0; edge < cfg.edges; ++edge) {
        std::uint32_t root = 0u;
        if (edge < in.dirty_per_cell)
            root = edge % 3u == 0u ? 0u : edge % 3u == 1u ? 4u : 6u;
        else {
            const std::array<std::uint32_t, 4> roots{{1u, 2u, 3u, 5u}};
            root = roots[(edge - in.dirty_per_cell) % roots.size()];
        }
        in.bindings.push_back({edge, root, 0u});
        in.digest = mix_hash(in.digest, edge);
        in.digest = mix_hash(in.digest, root);
    }
    in.program = {1u, in.nodes.data(), in.nodes.size(), in.bindings.data(),
        in.bindings.size(), cfg.edges, 4u, 0u};
    in.predicates.resize(cfg.cells * 4u);
    in.changed_predicates.resize(in.predicates.size());
    rng = cfg.seed ^ 0x94d049bb133111ebull;
    for (auto &predicate : in.predicates)
        predicate = unit_random(rng) < cfg.gate_density ? 1u : 0u;
    in.changed_predicates = in.predicates;
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell)
        in.changed_predicates[cell * 4u] ^= 1u;
    for (auto value : in.predicates) in.digest = mix_hash(in.digest, value);
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        in.dirty_offsets.push_back(in.dirty_edges.size());
        for (std::uint64_t ordinal = 0; ordinal < in.dirty_per_cell; ++ordinal)
            in.dirty_edges.push_back(ordinal);
    }
    in.dirty_offsets.push_back(in.dirty_edges.size());
    in.digest = mix_hash(in.digest, in.dirty_per_cell);
    for (auto ordinal : in.dirty_edges) in.digest = mix_hash(in.digest, ordinal);
    std::uint64_t gate_pass = 0;
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        const auto *p = in.predicates.data() + cell * 4u;
        for (const auto &binding : in.bindings) {
            const auto r = binding.root_node;
            const bool pass = r == 0u ? p[0] : r == 1u ? p[1] : r == 2u ? p[2]
                : r == 3u ? p[3] : r == 4u ? p[0] && p[1]
                : r == 5u ? p[2] || p[3] : p[0] && p[1] && p[2];
            gate_pass += pass;
        }
    }
    in.observed_gate_density = static_cast<double>(gate_pass)
        / static_cast<double>(cfg.cells * cfg.edges);
    for (const auto &term : in.primal_queries) {
        in.digest = mix_hash(in.digest, term.role);
        in.digest = mix_hash(in.digest, term.entity_id);
    }
    for (const auto &term : in.derivative_queries) {
        in.digest = mix_hash(in.digest, term.role);
        in.digest = mix_hash(in.digest, term.entity_id);
    }
    (void)f;
    return in;
}

bool matches_edge(const support::logical_support_edge_v1 &edge,
                  const support::logical_support_participant_v1 *participants,
                  const indexing::posting_term_v1 &term) {
    for (std::uint32_t i = 0; i < edge.participant_count; ++i) {
        const auto &p = participants[edge.participant_begin + i];
        if (p.role == term.role && p.entity_id == term.entity_id) return true;
    }
    return false;
}

void host_scan(const options &cfg, const fixture &f,
               const std::vector<indexing::posting_term_v1> &terms,
               host_candidates &out) {
    out.count = 0u;
    out.offsets[0] = 0u;
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        for (std::uint64_t edge = 0; edge < cfg.edges; ++edge) {
            if (!matches_edge(f.gpu_edges[edge], f.gpu_participants.data(), terms[cell])) continue;
            out.items[out.count++] = {f.gpu_edges[edge].logical_edge_id, edge};
        }
        out.offsets[cell + 1u] = out.count;
    }
}

void host_postings(const options &cfg,
                   const indexing::reverse_posting_view_v1 &index,
                   const std::vector<indexing::posting_term_v1> &terms,
                   std::vector<indexing::indexed_work_item_v1> &scratch,
                   host_candidates &out) {
    out.count = 0u;
    out.offsets[0] = 0u;
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        indexing::candidate_output_v1 output{out.items.data() + out.count,
            cfg.edges, 0u, 0u};
        const auto status = indexing::query_reverse_postings_v1(index,
            &terms[cell], 1u, indexing::posting_match_v1::any,
            {scratch.data(), scratch.size()}, &output);
        if (status != indexing::posting_status_v1::success)
            throw std::runtime_error("host posting query failed");
        out.count += output.written;
        out.offsets[cell + 1u] = out.count;
    }
}

struct host_gate {
    std::vector<std::uint64_t> bitmap;
    std::vector<std::uint8_t> node_scratch;
    std::vector<std::uint64_t> changed_bitmap;
    std::vector<std::uint8_t> changed_node_scratch;
    std::uint64_t words_per_cell = 0;
};

host_gate make_host_gate(const options &cfg, const inputs &in) {
    host_gate g;
    g.words_per_cell = cfg.edges / 64u + (cfg.edges % 64u != 0u);
    g.bitmap.resize(cfg.cells * g.words_per_cell);
    g.node_scratch.resize(in.nodes.size());
    const auto status = gating::evaluate_condition_bitmaps_v1(in.program, cfg.cells,
        in.predicates.data(), in.predicates.size(), g.bitmap.data(), g.bitmap.size(),
        g.node_scratch.data(), g.node_scratch.size());
    if (status != gating::condition_status_v1::success)
        throw std::runtime_error("host bitmap oracle failed");
    g.changed_bitmap.resize(cfg.cells * g.words_per_cell);
    g.changed_node_scratch.resize(in.nodes.size());
    const auto changed_status = gating::evaluate_condition_bitmaps_v1(in.program, cfg.cells,
        in.changed_predicates.data(), in.changed_predicates.size(), g.changed_bitmap.data(),
        g.changed_bitmap.size(), g.changed_node_scratch.data(), g.changed_node_scratch.size());
    if (changed_status != gating::condition_status_v1::success)
        throw std::runtime_error("host changed bitmap oracle failed");
    return g;
}

struct host_worksets {
    std::vector<std::uint64_t> primal_offsets;
    std::vector<std::uint64_t> derivative_offsets;
    std::vector<indexing::indexed_work_item_v1> primal_items;
    std::vector<indexing::indexed_work_item_v1> derivative_items;
    gating::gate_workset_pair_v1 pair{};
};

host_worksets make_host_worksets(const options &cfg) {
    host_worksets w;
    w.primal_offsets.resize(cfg.cells + 1u);
    w.derivative_offsets.resize(cfg.cells + 1u);
    w.primal_items.resize(cfg.cells * cfg.edges);
    w.derivative_items.resize(cfg.cells * cfg.edges * 2u);
    w.pair.primal = {w.primal_offsets.data(), w.primal_offsets.size(),
        w.primal_items.data(), w.primal_items.size(), 0u, 0u};
    w.pair.derivative = {w.derivative_offsets.data(), w.derivative_offsets.size(),
        w.derivative_items.data(), w.derivative_items.size(), 0u, 0u};
    return w;
}

void make_host_worksets_call(const options &cfg, const inputs &in,
                             const host_gate &gate,
                             const host_candidates &primal,
                             const host_candidates &derivative,
                             host_worksets &out) {
    const gating::per_cell_candidate_view_v1 p{primal.offsets.data(),
        primal.items.data(), primal.count};
    const gating::per_cell_candidate_view_v1 d{derivative.offsets.data(),
        derivative.items.data(), derivative.count};
    const auto status = gating::build_condition_worksets_v1(in.program, cfg.cells, 1u,
        gate.bitmap.data(), gate.bitmap.size(), p, d, &out.pair);
    if (status != gating::condition_status_v1::success)
        throw std::runtime_error("host workset oracle failed");
}

void compare_candidates(const host_candidates &a, const host_candidates &b,
                        const char *name) {
    if (a.count != b.count || a.offsets != b.offsets)
        throw std::runtime_error(std::string(name) + " offset/count mismatch");
    for (std::uint64_t i = 0; i < a.count; ++i)
        if (a.items[i].logical_edge_id != b.items[i].logical_edge_id
            || a.items[i].logical_edge_ordinal != b.items[i].logical_edge_ordinal)
            throw std::runtime_error(std::string(name) + " candidate identity mismatch");
}

__global__ void mark_rescan_flags_kernel(
    const support::logical_support_edge_v1 *edges,
    const support::logical_support_participant_v1 *participants,
    std::uint64_t edge_count, std::uint64_t cell_count,
    const std::uint64_t *query_offsets,
    const indexing::posting_term_v1 *primal_queries,
    const indexing::posting_term_v1 *derivative_queries,
    std::uint64_t *primal_flags, std::uint64_t *derivative_flags) {
    const auto flat = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total = edge_count * cell_count;
    if (flat >= total) return;
    const auto cell = flat / edge_count;
    const auto ordinal = flat - cell * edge_count;
    const auto edge = edges[ordinal];
    bool p_match = false, d_match = false;
    for (auto q = query_offsets[cell]; q < query_offsets[cell + 1u]; ++q) {
        const auto pt = primal_queries[q];
        const auto dt = derivative_queries[q];
        for (std::uint32_t slot = 0; slot < edge.participant_count; ++slot) {
            const auto participant = participants[edge.participant_begin + slot];
            p_match = p_match || (participant.role == pt.role
                && participant.entity_id == pt.entity_id);
            d_match = d_match || (participant.role == dt.role
                && participant.entity_id == dt.entity_id);
        }
    }
    primal_flags[flat] = p_match ? 1u : 0u;
    derivative_flags[flat] = d_match ? 1u : 0u;
}

__global__ void make_candidate_offsets_kernel(
    std::uint64_t cells, std::uint64_t edges,
    const std::uint64_t *p_flags, const std::uint64_t *p_prefix,
    const std::uint64_t *d_flags, const std::uint64_t *d_prefix,
    std::uint64_t *p_offsets, std::uint64_t *d_offsets) {
    const auto cell = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= cells) return;
    const auto total = cells * edges;
    const auto first = cell * edges;
    p_offsets[cell] = p_prefix[first];
    d_offsets[cell] = d_prefix[first];
    if (cell + 1u == cells) {
        p_offsets[cells] = p_prefix[total - 1u] + p_flags[total - 1u];
        d_offsets[cells] = d_prefix[total - 1u] + d_flags[total - 1u];
    }
}

__global__ void scatter_candidate_items_kernel(
    const support::logical_support_edge_v1 *edges,
    std::uint64_t edge_count, std::uint64_t cells,
    const std::uint64_t *p_flags, const std::uint64_t *p_prefix,
    const std::uint64_t *d_flags, const std::uint64_t *d_prefix,
    indexing::indexed_work_item_v1 *p_items,
    indexing::indexed_work_item_v1 *d_items) {
    const auto flat = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= edge_count * cells) return;
    const auto ordinal = flat % edge_count;
    const indexing::indexed_work_item_v1 item{edges[ordinal].logical_edge_id, ordinal};
    if (p_flags[flat]) p_items[p_prefix[flat]] = item;
    if (d_flags[flat]) d_items[d_prefix[flat]] = item;
}

struct timing_stats {
    std::vector<double> samples_ms;
    double p10 = 0, median = 0, p90 = 0, minimum = 0, maximum = 0;
};

double percentile(std::vector<double> sorted, double p) {
    std::sort(sorted.begin(), sorted.end());
    const double at = p * static_cast<double>(sorted.size() - 1u);
    const auto low = static_cast<std::size_t>(at);
    const auto high = std::min(low + 1u, sorted.size() - 1u);
    const double frac = at - low;
    return sorted[low] * (1.0 - frac) + sorted[high] * frac;
}

template <class Function>
timing_stats measure_cuda(cudaStream_t stream, std::uint32_t warmups,
                          std::uint32_t samples, std::uint32_t iterations,
                          Function &&function) {
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (std::uint32_t i = 0; i < warmups; ++i) function();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timing_stats stats;
    stats.samples_ms.reserve(samples);
    for (std::uint32_t sample = 0; sample < samples; ++sample) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (std::uint32_t i = 0; i < iterations; ++i) function();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        stats.samples_ms.push_back(milliseconds / iterations);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    stats.minimum = *std::min_element(stats.samples_ms.begin(), stats.samples_ms.end());
    stats.maximum = *std::max_element(stats.samples_ms.begin(), stats.samples_ms.end());
    stats.p10 = percentile(stats.samples_ms, 0.1);
    stats.median = percentile(stats.samples_ms, 0.5);
    stats.p90 = percentile(stats.samples_ms, 0.9);
    return stats;
}

timing_stats measure_host(std::uint32_t warmups, std::uint32_t samples,
                          const std::function<void()> &function) {
    for (std::uint32_t i = 0; i < warmups; ++i) function();
    timing_stats stats;
    stats.samples_ms.reserve(samples);
    for (std::uint32_t sample = 0; sample < samples; ++sample) {
        const auto start = host_clock::now();
        function();
        const auto stop = host_clock::now();
        stats.samples_ms.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
    }
    stats.minimum = *std::min_element(stats.samples_ms.begin(), stats.samples_ms.end());
    stats.maximum = *std::max_element(stats.samples_ms.begin(), stats.samples_ms.end());
    stats.p10 = percentile(stats.samples_ms, 0.1);
    stats.median = percentile(stats.samples_ms, 0.5);
    stats.p90 = percentile(stats.samples_ms, 0.9);
    return stats;
}

void print_stats(const char *name, const timing_stats &stats, bool comma) {
    std::cout << "\"" << name << "\":{\"min_ms\":" << stats.minimum
        << ",\"p10_ms\":" << stats.p10 << ",\"median_ms\":" << stats.median
        << ",\"p90_ms\":" << stats.p90 << ",\"max_ms\":" << stats.maximum
        << ",\"samples_ms\":[";
    for (std::size_t i = 0; i < stats.samples_ms.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << stats.samples_ms[i];
    }
    std::cout << "]}" << (comma ? "," : "");
}

struct device_context {
    cudaStream_t stream = nullptr;
    cudaDeviceProp properties{};
    int device = 0;
    std::string name;
    std::string uuid;
};

device_context make_device_context() {
    device_context c;
    CUDA_CHECK(cudaGetDevice(&c.device));
    CUDA_CHECK(cudaGetDeviceProperties(&c.properties, c.device));
    c.name = c.properties.name;
    char uuid[80]{};
    std::snprintf(uuid, sizeof(uuid), "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        static_cast<unsigned char>(c.properties.uuid.bytes[0]),
        static_cast<unsigned char>(c.properties.uuid.bytes[1]),
        static_cast<unsigned char>(c.properties.uuid.bytes[2]),
        static_cast<unsigned char>(c.properties.uuid.bytes[3]),
        static_cast<unsigned char>(c.properties.uuid.bytes[4]),
        static_cast<unsigned char>(c.properties.uuid.bytes[5]),
        static_cast<unsigned char>(c.properties.uuid.bytes[6]),
        static_cast<unsigned char>(c.properties.uuid.bytes[7]),
        static_cast<unsigned char>(c.properties.uuid.bytes[8]),
        static_cast<unsigned char>(c.properties.uuid.bytes[9]),
        static_cast<unsigned char>(c.properties.uuid.bytes[10]),
        static_cast<unsigned char>(c.properties.uuid.bytes[11]),
        static_cast<unsigned char>(c.properties.uuid.bytes[12]),
        static_cast<unsigned char>(c.properties.uuid.bytes[13]),
        static_cast<unsigned char>(c.properties.uuid.bytes[14]),
        static_cast<unsigned char>(c.properties.uuid.bytes[15]));
    c.uuid = uuid;
    CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
    return c;
}

void run(const options &cfg) {
    fixture f = make_fixture(cfg);
    inputs in = make_inputs(cfg, f);
    const auto requirements = indexing::query_reverse_posting_requirements_v1(f.snapshot);
    std::vector<indexing::reverse_posting_record_v1> posting_scratch(requirements.record_capacity);
    std::vector<indexing::reverse_posting_record_v1> host_entries(requirements.maximum_unique_entries);
    indexing::reverse_posting_view_v1 host_index{};
    auto build_host_index = [&] {
        if (indexing::build_reverse_postings_v1(f.snapshot, posting_scratch.data(),
                posting_scratch.size(), host_entries.data(), host_entries.size(), &host_index)
            != indexing::posting_status_v1::success)
            throw std::runtime_error("host reverse posting build failed");
    };
    const auto host_index_stats = measure_host(cfg.warmup, cfg.samples, build_host_index);
    build_host_index();
    const auto projection_req = indexing::cuda_posting_projection_requirements_v1(host_index);
    std::vector<indexing::posting_term_v1> projection_terms(projection_req.term_count);
    std::vector<std::uint64_t> projection_offsets(projection_req.term_count + 1u);
    std::vector<indexing::indexed_work_item_v1> projection_items(projection_req.item_count);
    indexing::cuda_posting_prepare_requirements_v1 projection_required{};
    auto build_projection = [&] {
        if (indexing::build_cuda_posting_projection_v1(host_index,
                projection_terms.data(), projection_terms.size(), projection_offsets.data(),
                projection_offsets.size(), projection_items.data(), projection_items.size(),
                &projection_required)
            != indexing::cuda_posting_prepare_status_v1::success)
            throw std::runtime_error("host CUDA posting projection build failed");
    };
    const auto projection_stats = measure_host(cfg.warmup, cfg.samples, build_projection);
    build_projection();

    device_context device = make_device_context();
    const std::uint64_t n = cfg.edges * cfg.cells;
    const std::uint64_t bitmap_words_per_cell = cfg.edges / 64u + (cfg.edges % 64u != 0u);
    const std::uint64_t bitmap_count = cfg.cells * bitmap_words_per_cell;
    device_buffer<support::logical_support_edge_v1> d_edges(f.gpu_edges.size());
    device_buffer<support::logical_support_participant_v1> d_participants(f.gpu_participants.size());
    device_buffer<indexing::posting_term_v1> d_index_terms(projection_terms.size());
    device_buffer<std::uint64_t> d_index_offsets(projection_offsets.size());
    device_buffer<indexing::indexed_work_item_v1> d_index_items(projection_items.size());
    device_buffer<std::uint64_t> d_query_offsets(in.query_offsets.size());
    device_buffer<indexing::posting_term_v1> d_primal_queries(in.primal_queries.size());
    device_buffer<indexing::posting_term_v1> d_derivative_queries(in.derivative_queries.size());
    device_buffer<std::uint8_t> d_predicates(in.predicates.size());
    device_buffer<std::uint8_t> d_changed_predicates(in.changed_predicates.size());
    device_buffer<gating::condition_node_v1> d_nodes(in.nodes.size());
    device_buffer<gating::edge_condition_binding_v1> d_bindings(in.bindings.size());
    device_buffer<std::uint64_t> d_dirty_offsets(in.dirty_offsets.size());
    device_buffer<std::uint64_t> d_dirty_edges(in.dirty_edges.size());
    device_buffer<std::uint64_t> d_bitmaps(bitmap_count);
    device_buffer<std::uint64_t> d_dirty_bitmaps(bitmap_count);
    device_buffer<std::uint8_t> d_node_scratch(cfg.cells * in.nodes.size());
    device_buffer<gating::cuda_workset_status_v1> d_gate_status(1u);

    // Device allocations are outside the steady-state phases. This captures
    // caller-reserved memory and separates projection setup from allocation.
    copy_to_device(d_edges, f.gpu_edges, device.stream);
    copy_to_device(d_participants, f.gpu_participants, device.stream);
    copy_to_device(d_index_terms, projection_terms, device.stream);
    copy_to_device(d_index_offsets, projection_offsets, device.stream);
    copy_to_device(d_index_items, projection_items, device.stream);
    copy_to_device(d_query_offsets, in.query_offsets, device.stream);
    copy_to_device(d_primal_queries, in.primal_queries, device.stream);
    copy_to_device(d_derivative_queries, in.derivative_queries, device.stream);
    copy_to_device(d_predicates, in.predicates, device.stream);
    copy_to_device(d_changed_predicates, in.changed_predicates, device.stream);
    copy_to_device(d_nodes, std::vector<gating::condition_node_v1>(in.nodes.begin(), in.nodes.end()), device.stream);
    copy_to_device(d_bindings, in.bindings, device.stream);
    copy_to_device(d_dirty_offsets, in.dirty_offsets, device.stream);
    copy_to_device(d_dirty_edges, in.dirty_edges, device.stream);

    indexing::cuda_reverse_posting_view_v1 device_index{
        host_index.relation_id, host_index.structure_id, host_index.structure_epoch,
        d_index_terms.data, d_index_offsets.data, projection_req.term_count,
        d_index_items.data, projection_req.item_count};
    gating::condition_program_view_v1 device_program{in.program.structure_epoch,
        d_nodes.data, in.program.node_count, d_bindings.data,
        in.program.binding_count, in.program.edge_count, in.program.predicate_count, 0u};

    // Device posting workspaces and outputs.
    device_buffer<indexing::cuda_posting_scratch_v1::cursor_v1> d_p_cursors(cfg.cells);
    device_buffer<indexing::cuda_posting_scratch_v1::cursor_v1> d_d_cursors(cfg.cells);
    device_buffer<std::uint64_t> d_p_counts(cfg.cells), d_d_counts(cfg.cells);
    device_buffer<indexing::cuda_posting_status_v1> d_p_post_status(1u), d_d_post_status(1u);
    device_buffer<std::uint64_t> d_p_post_offsets(cfg.cells + 1u), d_d_post_offsets(cfg.cells + 1u);
    device_buffer<indexing::indexed_work_item_v1> d_p_post_items(n), d_d_post_items(n);
    indexing::cuda_posting_scratch_v1 p_post_scratch{
        d_p_cursors.data, cfg.cells, d_p_counts.data, d_p_post_status.data};
    indexing::cuda_posting_scratch_v1 d_post_scratch{
        d_d_cursors.data, cfg.cells, d_d_counts.data, d_d_post_status.data};
    indexing::cuda_posting_output_v1 p_post_output{
        d_p_post_offsets.data, cfg.cells + 1u, d_p_post_items.data, n};
    indexing::cuda_posting_output_v1 d_post_output{
        d_d_post_offsets.data, cfg.cells + 1u, d_d_post_items.data, n};

    // Full-scan baseline workspaces and CUB exclusive-scan scratch.
    device_buffer<std::uint64_t> d_p_flags(n), d_d_flags(n), d_p_prefix(n), d_d_prefix(n);
    device_buffer<std::uint64_t> d_p_scan_offsets(cfg.cells + 1u), d_d_scan_offsets(cfg.cells + 1u);
    device_buffer<indexing::indexed_work_item_v1> d_p_scan_items(n), d_d_scan_items(n);
    std::size_t p_scan_bytes = 0, d_scan_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, p_scan_bytes,
        d_p_flags.data, d_p_prefix.data, n, device.stream));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, d_scan_bytes,
        d_d_flags.data, d_d_prefix.data, n, device.stream));
    device_buffer<std::uint8_t> d_scan_temp(std::max(p_scan_bytes, d_scan_bytes));

    auto full_scan_device = [&] {
        constexpr unsigned threads = 256u;
        const auto grid = static_cast<unsigned>((n + threads - 1u) / threads);
        mark_rescan_flags_kernel<<<grid, threads, 0, device.stream>>>(
            d_edges.data, d_participants.data, cfg.edges, cfg.cells,
            d_query_offsets.data, d_primal_queries.data, d_derivative_queries.data,
            d_p_flags.data, d_d_flags.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_scan_temp.data, p_scan_bytes,
            d_p_flags.data, d_p_prefix.data, n, device.stream));
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_scan_temp.data, d_scan_bytes,
            d_d_flags.data, d_d_prefix.data, n, device.stream));
        make_candidate_offsets_kernel<<<static_cast<unsigned>((cfg.cells + threads - 1u) / threads),
            threads, 0, device.stream>>>(cfg.cells, cfg.edges,
            d_p_flags.data, d_p_prefix.data, d_d_flags.data, d_d_prefix.data,
            d_p_scan_offsets.data, d_d_scan_offsets.data);
        CUDA_CHECK(cudaGetLastError());
        scatter_candidate_items_kernel<<<grid, threads, 0, device.stream>>>(
            d_edges.data, cfg.edges, cfg.cells,
            d_p_flags.data, d_p_prefix.data, d_d_flags.data, d_d_prefix.data,
            d_p_scan_items.data, d_d_scan_items.data);
        CUDA_CHECK(cudaGetLastError());
    };

    auto postings_device = [&] {
        CUDA_CHECK(indexing::cuda_query_reverse_postings_any_v1(device_index,
            in.program.structure_epoch, cfg.cells, d_query_offsets.data,
            d_primal_queries.data, in.primal_queries.size(), p_post_output,
            p_post_scratch, device.stream));
        CUDA_CHECK(indexing::cuda_query_reverse_postings_any_v1(device_index,
            in.program.structure_epoch, cfg.cells, d_query_offsets.data,
            d_derivative_queries.data, in.derivative_queries.size(), d_post_output,
            d_post_scratch, device.stream));
    };

    auto evaluate_full_bitmap = [&] {
        CUDA_CHECK(gating::cuda_evaluate_condition_bitmaps_v1(device_program,
            cfg.cells, d_predicates.data, in.predicates.size(), d_bitmaps.data,
            bitmap_count, d_node_scratch.data, d_node_scratch.size,
            d_gate_status.data, device.stream));
    };

    // Gate worksets reserve at most E items/cell for primal, and E+E for the
    // derivative eligible union. Offsets are fixed-capacity stage boundaries.
    device_buffer<std::uint64_t> d_p_stage_offsets(cfg.cells + 1u), d_d_stage_offsets(cfg.cells + 1u);
    device_buffer<indexing::indexed_work_item_v1> d_p_stage_items(n), d_d_stage_items(n * 2u);
    device_buffer<std::uint64_t> d_p_counts_work(cfg.cells), d_d_counts_work(cfg.cells);
    device_buffer<std::uint64_t> d_p_work_offsets(cfg.cells + 1u), d_d_work_offsets(cfg.cells + 1u);
    device_buffer<indexing::indexed_work_item_v1> d_p_work_items(n), d_d_work_items(n * 2u);
    std::vector<std::uint64_t> p_stage_offsets(cfg.cells + 1u), d_stage_offsets(cfg.cells + 1u);
    for (std::uint64_t c = 0; c < cfg.cells; ++c) {
        p_stage_offsets[c] = c * cfg.edges;
        d_stage_offsets[c] = c * cfg.edges * 2u;
    }
    p_stage_offsets[cfg.cells] = n;
    d_stage_offsets[cfg.cells] = n * 2u;
    copy_to_device(d_p_stage_offsets, p_stage_offsets, device.stream);
    copy_to_device(d_d_stage_offsets, d_stage_offsets, device.stream);
    gating::cuda_workset_scratch_v1 work_scratch{};
    work_scratch.primal_offsets = d_p_stage_offsets.data;
    work_scratch.primal_items = d_p_stage_items.data;
    work_scratch.primal_item_capacity = n;
    work_scratch.derivative_offsets = d_d_stage_offsets.data;
    work_scratch.derivative_items = d_d_stage_items.data;
    work_scratch.derivative_item_capacity = n * 2u;
    work_scratch.primal_counts = d_p_counts_work.data;
    work_scratch.derivative_counts = d_d_counts_work.data;
    work_scratch.status = d_gate_status.data;
    work_scratch.primal_offsets_capacity = cfg.cells + 1u;
    work_scratch.derivative_offsets_capacity = cfg.cells + 1u;
    gating::cuda_output_csr_v1 p_work_output{
        d_p_work_offsets.data, cfg.cells + 1u, d_p_work_items.data, n};
    gating::cuda_output_csr_v1 d_work_output{
        d_d_work_offsets.data, cfg.cells + 1u, d_d_work_items.data, n * 2u};
    auto gate_scan_worksets = [&] {
        // item_count is the accessible buffer capacity here. The CSR final
        // offset delimits the populated prefix; this keeps the pipeline
        // device-resident without a host count synchronization.
        const gating::cuda_candidate_csr_v1 p{d_p_scan_offsets.data, d_p_scan_items.data, n};
        const gating::cuda_candidate_csr_v1 d{d_d_scan_offsets.data, d_d_scan_items.data, n};
        CUDA_CHECK(gating::cuda_build_condition_worksets_v1(device_program, cfg.cells,
            d_bitmaps.data, bitmap_count, p, d, p_work_output, d_work_output,
            work_scratch, device.stream));
    };
    auto gate_post_worksets = [&] {
        const gating::cuda_candidate_csr_v1 p{d_p_post_offsets.data, d_p_post_items.data, n};
        const gating::cuda_candidate_csr_v1 d{d_d_post_offsets.data, d_d_post_items.data, n};
        CUDA_CHECK(gating::cuda_build_condition_worksets_v1(device_program, cfg.cells,
            d_bitmaps.data, bitmap_count, p, d, p_work_output, d_work_output,
            work_scratch, device.stream));
    };
    auto full_pipeline = [&] {
        full_scan_device();
        evaluate_full_bitmap();
        gate_scan_worksets();
    };
    auto posting_pipeline = [&] {
        postings_device();
        evaluate_full_bitmap();
        gate_post_worksets();
    };

    // Upload projection and all immutable inputs once, then profile upload
    // alone with repeated copies into caller-reserved device storage.
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    const auto projection_upload_stats = measure_cuda(device.stream, cfg.warmup,
        cfg.samples, 1u, [&] {
            copy_to_device(d_index_terms, projection_terms, device.stream);
            copy_to_device(d_index_offsets, projection_offsets, device.stream);
            copy_to_device(d_index_items, projection_items, device.stream);
        });
    // Re-upload all non-projection fixture inputs before exact validation.
    copy_to_device(d_edges, f.gpu_edges, device.stream);
    copy_to_device(d_participants, f.gpu_participants, device.stream);
    copy_to_device(d_query_offsets, in.query_offsets, device.stream);
    copy_to_device(d_primal_queries, in.primal_queries, device.stream);
    copy_to_device(d_derivative_queries, in.derivative_queries, device.stream);
    copy_to_device(d_predicates, in.predicates, device.stream);
    copy_to_device(d_changed_predicates, in.changed_predicates, device.stream);
    copy_to_device(d_nodes, std::vector<gating::condition_node_v1>(in.nodes.begin(), in.nodes.end()), device.stream);
    copy_to_device(d_bindings, in.bindings, device.stream);
    copy_to_device(d_dirty_offsets, in.dirty_offsets, device.stream);
    copy_to_device(d_dirty_edges, in.dirty_edges, device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));

    // Independent CPU scan/postings/gating oracle before timing.
    host_candidates expected_p = make_host_candidates(cfg), expected_d = make_host_candidates(cfg);
    host_candidates expected_post_p = make_host_candidates(cfg), expected_post_d = make_host_candidates(cfg);
    host_scan(cfg, f, in.primal_queries, expected_p);
    host_scan(cfg, f, in.derivative_queries, expected_d);
    std::vector<indexing::indexed_work_item_v1> host_query_scratch(
        std::max<std::uint64_t>(1u, host_index.entry_count));
    host_postings(cfg, host_index, in.primal_queries, host_query_scratch, expected_post_p);
    host_postings(cfg, host_index, in.derivative_queries, host_query_scratch, expected_post_d);
    compare_candidates(expected_p, expected_post_p, "host scan/posting primal");
    compare_candidates(expected_d, expected_post_d, "host scan/posting derivative");
    host_gate expected_gate = make_host_gate(cfg, in);
    host_worksets expected_scan_work = make_host_worksets(cfg);
    host_worksets expected_post_work = make_host_worksets(cfg);
    make_host_worksets_call(cfg, in, expected_gate, expected_p, expected_d, expected_scan_work);
    make_host_worksets_call(cfg, in, expected_gate, expected_post_p, expected_post_d, expected_post_work);

    // Device correctness for the exhaustive baseline and the exact index.
    full_scan_device();
    postings_device();
    evaluate_full_bitmap();
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    const auto scan_p_offsets = copy_from_device(d_p_scan_offsets.data, cfg.cells + 1u);
    const auto scan_d_offsets = copy_from_device(d_d_scan_offsets.data, cfg.cells + 1u);
    const auto post_p_offsets = copy_from_device(d_p_post_offsets.data, cfg.cells + 1u);
    const auto post_d_offsets = copy_from_device(d_d_post_offsets.data, cfg.cells + 1u);
    host_candidates device_scan_p{scan_p_offsets,
        copy_from_device(d_p_scan_items.data, scan_p_offsets.back()), scan_p_offsets.back()};
    host_candidates device_scan_d{scan_d_offsets,
        copy_from_device(d_d_scan_items.data, scan_d_offsets.back()), scan_d_offsets.back()};
    host_candidates device_post_p{post_p_offsets,
        copy_from_device(d_p_post_items.data, post_p_offsets.back()), post_p_offsets.back()};
    host_candidates device_post_d{post_d_offsets,
        copy_from_device(d_d_post_items.data, post_d_offsets.back()), post_d_offsets.back()};
    compare_candidates(expected_p, device_scan_p, "CUDA full scan primal");
    compare_candidates(expected_d, device_scan_d, "CUDA full scan derivative");
    compare_candidates(expected_p, device_post_p, "CUDA postings primal");
    compare_candidates(expected_d, device_post_d, "CUDA postings derivative");
    indexing::cuda_posting_status_v1 p_post_status{}, d_post_status{};
    CUDA_CHECK(cudaMemcpy(&p_post_status, d_p_post_status.data, sizeof(p_post_status), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&d_post_status, d_d_post_status.data, sizeof(d_post_status), cudaMemcpyDeviceToHost));
    if (p_post_status.code != 0 || p_post_status.required != expected_p.count
        || d_post_status.code != 0 || d_post_status.required != expected_d.count)
        throw std::runtime_error("CUDA posting status/count oracle mismatch");
    const auto device_bitmap = copy_from_device(d_bitmaps.data, bitmap_count);
    if (device_bitmap != expected_gate.bitmap)
        throw std::runtime_error("CUDA bitmap differs from host oracle");

    // Dirty updates are compared at the same full-batch state scope.
    copy_to_device(d_dirty_bitmaps, expected_gate.bitmap, device.stream);
    CUDA_CHECK(gating::cuda_update_condition_bitmaps_v1(device_program, cfg.cells,
        d_changed_predicates.data, in.changed_predicates.size(), d_dirty_offsets.data,
        d_dirty_edges.data, in.dirty_edges.size(), d_dirty_bitmaps.data, bitmap_count,
        d_node_scratch.data, d_node_scratch.size, d_gate_status.data, device.stream));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    const auto dirty_bitmap = copy_from_device(d_dirty_bitmaps.data, bitmap_count);
    if (dirty_bitmap != expected_gate.changed_bitmap)
        throw std::runtime_error("CUDA dirty bitmap differs from full host rebuild");

    gate_scan_worksets();
    gate_post_worksets();
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    const auto work_status_after_post = [&] {
        gating::cuda_workset_status_v1 s{};
        CUDA_CHECK(cudaMemcpy(&s, d_gate_status.data, sizeof(s), cudaMemcpyDeviceToHost));
        return s;
    }();
    if (work_status_after_post.code != 0
        || work_status_after_post.primal_required != expected_post_work.pair.primal.written
        || work_status_after_post.derivative_required != expected_post_work.pair.derivative.written)
        throw std::runtime_error("CUDA gate workset required-count mismatch");
    auto verify_worksets = [&](const host_worksets &expected,
                               const device_buffer<std::uint64_t> &p_offsets,
                               const device_buffer<std::uint64_t> &d_offsets,
                               const device_buffer<indexing::indexed_work_item_v1> &p_items,
                               const device_buffer<indexing::indexed_work_item_v1> &d_items,
                               const char *label) {
        const auto po = copy_from_device(p_offsets.data, cfg.cells + 1u);
        const auto d_o = copy_from_device(d_offsets.data, cfg.cells + 1u);
        if (!std::equal(po.begin(), po.end(), expected.pair.primal.offsets)
            || !std::equal(d_o.begin(), d_o.end(), expected.pair.derivative.offsets))
            throw std::runtime_error(std::string(label) + " offsets differ from host oracle");
        const auto p = copy_from_device(p_items.data, po.back());
        const auto d = copy_from_device(d_items.data, d_o.back());
        for (std::uint64_t i = 0; i < po.back(); ++i)
            if (p[i].logical_edge_id != expected.pair.primal.items[i].logical_edge_id
                || p[i].logical_edge_ordinal != expected.pair.primal.items[i].logical_edge_ordinal)
                throw std::runtime_error(std::string(label) + " primal values differ");
        for (std::uint64_t i = 0; i < d_o.back(); ++i)
            if (d[i].logical_edge_id != expected.pair.derivative.items[i].logical_edge_id
                || d[i].logical_edge_ordinal != expected.pair.derivative.items[i].logical_edge_ordinal)
                throw std::runtime_error(std::string(label) + " derivative values differ");
    };
    verify_worksets(expected_scan_work, d_p_work_offsets, d_d_work_offsets,
                    d_p_work_items, d_d_work_items, "CUDA scan workset");
    gate_post_worksets();
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    verify_worksets(expected_post_work, d_p_work_offsets, d_d_work_offsets,
                    d_p_work_items, d_d_work_items, "CUDA postings workset");

    const auto iterations = cfg.iterations;
    const auto scan_discovery_stats = measure_cuda(device.stream, cfg.warmup,
        cfg.samples, iterations, full_scan_device);
    const auto posting_discovery_stats = measure_cuda(device.stream, cfg.warmup,
        cfg.samples, iterations, postings_device);
    const auto bitmap_full_stats = measure_cuda(device.stream, cfg.warmup,
        cfg.samples, iterations, evaluate_full_bitmap);
    const auto bitmap_dirty_stats = measure_cuda(device.stream, cfg.warmup,
        cfg.samples, iterations, [&] {
            CUDA_CHECK(gating::cuda_update_condition_bitmaps_v1(device_program, cfg.cells,
                d_changed_predicates.data, in.changed_predicates.size(), d_dirty_offsets.data,
                d_dirty_edges.data, in.dirty_edges.size(), d_dirty_bitmaps.data, bitmap_count,
                d_node_scratch.data, d_node_scratch.size, d_gate_status.data, device.stream));
        });
    const auto scan_workset_stats = measure_cuda(device.stream, cfg.warmup,
        cfg.samples, iterations, gate_scan_worksets);
    const auto post_workset_stats = measure_cuda(device.stream, cfg.warmup,
        cfg.samples, iterations, gate_post_worksets);
    const auto scan_pipeline_stats = measure_cuda(device.stream, cfg.warmup,
        cfg.samples, iterations, full_pipeline);
    const auto post_pipeline_stats = measure_cuda(device.stream, cfg.warmup,
        cfg.samples, iterations, posting_pipeline);

    // A separate host-clock measurement includes host projection packing,
    // asynchronous transfer, and stream completion; device allocations remain
    // caller-reserved and are reported separately.
    const auto setup_stats = measure_host(cfg.warmup, cfg.samples, [&] {
        build_host_index();
        build_projection();
        copy_to_device(d_index_terms, projection_terms, device.stream);
        copy_to_device(d_index_offsets, projection_offsets, device.stream);
        copy_to_device(d_index_items, projection_items, device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
    });

    // Both worksets are sorted by stable edge ID, so count derivative-only
    // items with a linear merge rather than a quadratic membership scan.
    std::uint64_t derivative_only = 0;
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        const auto pb = expected_post_work.pair.primal.offsets[cell];
        const auto pe = expected_post_work.pair.primal.offsets[cell + 1u];
        const auto db = expected_post_work.pair.derivative.offsets[cell];
        const auto de = expected_post_work.pair.derivative.offsets[cell + 1u];
        auto p = pb;
        for (auto d = db; d < de; ++d) {
            const auto derivative_id = expected_post_work.pair.derivative.items[d].logical_edge_id;
            while (p < pe && expected_post_work.pair.primal.items[p].logical_edge_id < derivative_id)
                ++p;
            derivative_only += p == pe
                || expected_post_work.pair.primal.items[p].logical_edge_id != derivative_id;
        }
    }

    const auto structure_bytes =
        f.gpu_edges.size() * sizeof(support::logical_support_edge_v1)
        + f.gpu_participants.size() * sizeof(support::logical_support_participant_v1);
    const auto projection_bytes =
        projection_terms.size() * sizeof(indexing::posting_term_v1)
        + projection_offsets.size() * sizeof(std::uint64_t)
        + projection_items.size() * sizeof(indexing::indexed_work_item_v1);
    const auto posting_workspace_bytes =
        2u * cfg.cells * sizeof(indexing::cuda_posting_scratch_v1::cursor_v1)
        + 2u * cfg.cells * sizeof(std::uint64_t)
        + 2u * sizeof(indexing::cuda_posting_status_v1)
        + 2u * (cfg.cells + 1u) * sizeof(std::uint64_t)
        + 2u * n * sizeof(indexing::indexed_work_item_v1);
    const auto rescan_workspace_bytes =
        4u * n * sizeof(std::uint64_t)
        + 2u * (cfg.cells + 1u) * sizeof(std::uint64_t)
        + 2u * n * sizeof(indexing::indexed_work_item_v1)
        + std::max(p_scan_bytes, d_scan_bytes);
    const auto workset_workspace_bytes =
        4u * (cfg.cells + 1u) * sizeof(std::uint64_t)
        + 3u * n * sizeof(indexing::indexed_work_item_v1)
        + 2u * cfg.cells * sizeof(std::uint64_t)
        + 3u * n * sizeof(indexing::indexed_work_item_v1);
    const auto state_and_gate_bytes =
        (cfg.cells + 1u) * sizeof(std::uint64_t)
        + 2u * cfg.cells * sizeof(indexing::posting_term_v1)
        + 2u * in.predicates.size() * sizeof(std::uint8_t)
        + in.nodes.size() * sizeof(gating::condition_node_v1)
        + in.bindings.size() * sizeof(gating::edge_condition_binding_v1)
        + (cfg.cells + 1u) * sizeof(std::uint64_t)
        + in.dirty_edges.size() * sizeof(std::uint64_t)
        + 2u * bitmap_count * sizeof(std::uint64_t)
        + cfg.cells * in.nodes.size() * sizeof(std::uint8_t)
        + sizeof(gating::cuda_workset_status_v1);
    const auto total_device_bytes = total_requested_device_bytes();
    const auto accounted_device_bytes = structure_bytes + projection_bytes
        + posting_workspace_bytes + rescan_workspace_bytes
        + workset_workspace_bytes + state_and_gate_bytes;
    if (accounted_device_bytes != total_device_bytes)
        throw std::runtime_error("device allocation accounting mismatch");

    std::cout << std::setprecision(9)
        << "{\"schema\":\"cellerator.sparse_support.cuda_index_bench.v1\","
        << "\"scope\":\"cuda_device_resident_and_setup_separate\","
        << "\"case_id\":\"" << cfg.case_id << "\",\"correctness\":\"exact_host_oracle_pass\","
        << "\"fixture\":{\"edges\":" << cfg.edges << ",\"cells\":" << cfg.cells
        << ",\"entities\":" << cfg.entities << ",\"delta_edges\":" << f.delta.edges.size()
        << ",\"participant_roles\":3,\"observed_gate_density\":" << in.observed_gate_density
        << ",\"source_skew\":" << cfg.source_skew << ",\"dirty_fraction\":" << cfg.dirty_fraction
        << ",\"dirty_edges_per_cell\":" << in.dirty_per_cell << ",\"seed\":" << cfg.seed
        << ",\"digest\":\"0x" << std::hex << mix_hash(f.digest, in.digest) << std::dec << "\"},"
        << "\"counts\":{\"primal_candidates\":" << expected_p.count
        << ",\"derivative_candidates\":" << expected_d.count
        << ",\"primal_work_items\":" << expected_post_work.pair.primal.written
        << ",\"derivative_union_work_items\":" << expected_post_work.pair.derivative.written
        << ",\"derivative_only_work_items\":" << derivative_only
        << ",\"postings_terms\":" << projection_req.term_count
        << ",\"postings_records\":" << projection_req.item_count
        << ",\"warmup\":" << cfg.warmup << ",\"samples\":" << cfg.samples
        << ",\"iterations_per_sample\":" << iterations << "},"
        << "\"memory_bytes\":{\"device_allocations_requested\":" << total_device_bytes
        << ",\"structure\":" << structure_bytes
        << ",\"posting_projection\":" << projection_bytes
        << ",\"posting_workspace_and_outputs\":" << posting_workspace_bytes
        << ",\"full_rescan_workspace_and_outputs\":" << rescan_workspace_bytes
        << ",\"condition_and_query_state\":" << state_and_gate_bytes
        << ",\"gate_workset_workspace_and_outputs\":" << workset_workspace_bytes
        << "},\"timings_ms\":{";
    print_stats("host_posting_index_build", host_index_stats, true);
    print_stats("host_device_projection_prepare", projection_stats, true);
    print_stats("posting_setup_projection_h2d_stream_complete", setup_stats, true);
    print_stats("projection_h2d_only", projection_upload_stats, true);
    print_stats("cuda_full_rescan_pair_discovery", scan_discovery_stats, true);
    print_stats("cuda_reverse_posting_pair_discovery", posting_discovery_stats, true);
    print_stats("cuda_condition_bitmap_full_batch", bitmap_full_stats, true);
    print_stats("cuda_condition_bitmap_dirty_batch", bitmap_dirty_stats, true);
    print_stats("cuda_gate_worksets_after_rescan", scan_workset_stats, true);
    print_stats("cuda_gate_worksets_after_postings", post_workset_stats, true);
    print_stats("cuda_complete_rescan_bitmap_workset", scan_pipeline_stats, true);
    print_stats("cuda_complete_postings_bitmap_workset", post_pipeline_stats, false);
    std::cout << "},\"reuse_amortization_ms\":{\"postings_setup_plus_pipeline_reuse_1\":"
        << setup_stats.median + post_pipeline_stats.median
        << ",\"postings_setup_plus_pipeline_reuse_16\":"
        << setup_stats.median / 16.0 + post_pipeline_stats.median << "},"
        << "\"device\":{\"name\":\"" << device.name << "\",\"uuid\":\""
        << device.uuid << "\",\"compute_capability\":" << device.properties.major
        << '.' << device.properties.minor << ",\"total_memory_bytes\":"
        << device.properties.totalGlobalMem << "},\"timing_policy\":{"
        << "\"steady_state_resident\":true,\"h2d_in_steady_state\":false,"
        << "\"cuda_events\":true,\"host_affinity_or_clock_control\":false}}\n";
    CUDA_CHECK(cudaStreamDestroy(device.stream));
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto cfg = parse_options(argc, argv);
        run(cfg);
    } catch (const std::exception &e) {
        std::cerr << "cuda_indexing_bench: " << e.what() << '\n';
        return 2;
    }
    return 0;
}
