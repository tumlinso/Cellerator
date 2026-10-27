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
#include <fstream>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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
    std::string fixture_dir;
    bool host_oracle_only = false;
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
        else if (arg == "--fixture-dir") result.fixture_dir = value(i, "--fixture-dir");
        else if (arg == "--host-oracle-only") result.host_oracle_only = true;
        else if (arg == "--warmup") result.warmup = static_cast<std::uint32_t>(parse_u64(value(i, "--warmup"), "warmup"));
        else if (arg == "--samples") result.samples = static_cast<std::uint32_t>(parse_u64(value(i, "--samples"), "samples"));
        else if (arg == "--iterations") result.iterations = static_cast<std::uint32_t>(parse_u64(value(i, "--iterations"), "iterations"));
        else if (arg == "--help") {
            std::cout << "cuda_indexing_bench --case-id ID --edges N --cells N "
                         "--entities N --gate-density P --source-skew P "
                         "--dirty-fraction P --warmup N --samples N --iterations N "
                         "[--fixture-dir PATH] [--host-oracle-only]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown option: " + arg);
    }
    if (result.edges == 0 || result.cells == 0 || result.cells > 4096
        || result.entities < 3 || result.warmup > 100
        || result.samples < (result.host_oracle_only ? 1u : 3u)
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
        + device_buffer<std::uint32_t>::requested_bytes
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
    std::vector<std::uint64_t> query_offsets;
    std::vector<std::uint64_t> query_entities;
    std::vector<std::uint64_t> edge_sources;
    std::vector<std::uint32_t> edge_modules;
    std::vector<std::uint64_t> observed_offsets;
    std::vector<std::uint64_t> observed_ids;
    std::vector<std::uint8_t> observed_values;
    std::vector<std::uint8_t> latent_values;
    std::vector<std::uint32_t> local_change_cells;
    std::vector<std::uint64_t> local_change_genes;
    std::vector<std::uint8_t> local_change_values;
    std::vector<std::uint32_t> latent_change_groups;
    std::vector<std::uint32_t> latent_change_modules;
    std::vector<std::uint64_t> cell_query_candidate_counts;
    std::uint64_t module_count = 0u;
    bool biological_conditions = false;
    std::uint64_t digest = 1469598103934665603ull;
};

template <class T> std::vector<T> read_raw_array(const std::filesystem::path &directory,
                                                  const char *name,
                                                  std::uint64_t count) {
    const auto path = directory / (std::string(name) + ".bin");
    const auto expected = count * sizeof(T);
    if (count != 0u && expected / sizeof(T) != count)
        throw std::runtime_error("raw fixture array size overflow");
    if (!std::filesystem::is_regular_file(path)
        || std::filesystem::file_size(path) != expected)
        throw std::runtime_error(std::string("fixture array size mismatch: ") + path.string());
    std::vector<T> values(static_cast<std::size_t>(count));
    std::ifstream stream(path, std::ios::binary);
    if (!stream || (expected != 0u
        && !stream.read(reinterpret_cast<char *>(values.data()),
                        static_cast<std::streamsize>(expected))))
        throw std::runtime_error(std::string("could not read fixture array: ") + path.string());
    return values;
}

std::vector<std::uint64_t> read_raw_u64_array(const std::filesystem::path &directory,
                                              const char *name) {
    const auto path = directory / (std::string(name) + ".bin");
    if (!std::filesystem::is_regular_file(path)
        || std::filesystem::file_size(path) % sizeof(std::uint64_t) != 0u)
        throw std::runtime_error(std::string("fixture array has invalid byte size: ") + path.string());
    return read_raw_array<std::uint64_t>(directory, name,
        std::filesystem::file_size(path) / sizeof(std::uint64_t));
}

template <class T> std::vector<T> read_raw_vector(const std::filesystem::path &directory,
                                                 const char *name) {
    const auto path = directory / (std::string(name) + ".bin");
    if (!std::filesystem::is_regular_file(path)
        || std::filesystem::file_size(path) % sizeof(T) != 0u)
        throw std::runtime_error(std::string("fixture array has invalid byte size: ") + path.string());
    return read_raw_array<T>(directory, name,
        std::filesystem::file_size(path) / sizeof(T));
}

std::uint64_t choose_source(std::uint64_t &rng, const options &cfg) {
    if (unit_random(rng) < cfg.source_skew) return 1u;
    return 1u + random_next(rng) % cfg.entities;
}

fixture make_fixture(const options &cfg) {
    if (!cfg.fixture_dir.empty()) {
        const std::filesystem::path directory(cfg.fixture_dir);
        if (!std::filesystem::is_regular_file(directory / "manifest.json"))
            throw std::runtime_error("fixture bundle has no manifest.json");
        const auto edge_ids = read_raw_array<std::uint64_t>(directory, "edge_ids", cfg.edges);
        const auto sources = read_raw_array<std::int64_t>(
            directory, "edge_observed_predicate_id", cfg.edges);
        const auto destinations = read_raw_array<std::int64_t>(
            directory, "destination_ids", cfg.edges);
        const auto modules = read_raw_array<std::uint32_t>(directory, "edge_module_id", cfg.edges);
        const auto observed_predicate_ids = read_raw_array<std::int64_t>(
            directory, "edge_observed_predicate_id", cfg.edges);
        const auto latent_predicate_ids = read_raw_array<std::int64_t>(
            directory, "edge_latent_predicate_id", cfg.edges);
        const auto condition_indptr = read_raw_array<std::uint64_t>(
            directory, "edge_condition_indptr", cfg.edges + 1u);
        const auto condition_ids = read_raw_array<std::int64_t>(
            directory, "edge_condition_predicate_id", cfg.edges * 2u);
        fixture f;
        f.edge_sources.resize(cfg.edges);
        f.edge_modules = modules;
        f.module_count = static_cast<std::uint64_t>(
            *std::max_element(modules.begin(), modules.end())) + 1u;
        f.observed_offsets = read_raw_array<std::uint64_t>(directory,
            "cell_observed_predicate_offsets", cfg.cells + 1u);
        f.observed_ids = read_raw_u64_array(directory, "cell_observed_predicate_ids");
        f.observed_values = read_raw_array<std::uint8_t>(directory,
            "cell_observed_predicate_values", f.observed_ids.size());
        f.latent_values = read_raw_array<std::uint8_t>(directory,
            "gate_initial_state", cfg.cells * f.module_count);
        f.local_change_cells = read_raw_vector<std::uint32_t>(directory,
            "local_gene_change_cell_id");
        f.local_change_genes = read_raw_u64_array(directory, "local_gene_change_gene_id");
        f.local_change_values = read_raw_array<std::uint8_t>(directory,
            "local_gene_change_value", f.local_change_genes.size());
        f.latent_change_groups = read_raw_vector<std::uint32_t>(directory,
            std::filesystem::is_regular_file(directory / "gate_change_group_id.bin")
                ? "gate_change_group_id" : "cell_cohort_id");
        f.latent_change_modules = read_raw_vector<std::uint32_t>(directory,
            "gate_change_predicate_id");
        f.cell_query_candidate_counts = read_raw_array<std::uint64_t>(directory,
            "cell_query_candidate_counts", cfg.cells);
        f.biological_conditions = true;
        f.base.edges.reserve(cfg.edges);
        f.base.participants.reserve(cfg.edges * 2u);
        std::vector<std::pair<std::int64_t, std::int64_t>> endpoint_pairs;
        endpoint_pairs.reserve(cfg.edges);
        std::vector<std::uint64_t> expected_source_counts(cfg.entities, 0u);
        for (std::uint64_t ordinal = 0; ordinal < cfg.edges; ++ordinal) {
            if (edge_ids[ordinal] == 0u || sources[ordinal] < 0
                || destinations[ordinal] < 0
                || static_cast<std::uint64_t>(sources[ordinal]) >= cfg.entities
                || static_cast<std::uint64_t>(destinations[ordinal]) >= cfg.entities
                || sources[ordinal] == destinations[ordinal]
                || modules[ordinal] >= f.module_count
                || observed_predicate_ids[ordinal] != sources[ordinal]
                || latent_predicate_ids[ordinal]
                    != static_cast<std::int64_t>(cfg.entities + modules[ordinal])
                || condition_indptr[ordinal] != ordinal * 2u
                || condition_indptr[ordinal + 1u] != (ordinal + 1u) * 2u
                || condition_ids[ordinal * 2u] != sources[ordinal]
                || condition_ids[ordinal * 2u + 1u] != latent_predicate_ids[ordinal]
                || (ordinal != 0u && edge_ids[ordinal] <= edge_ids[ordinal - 1u]))
                throw std::runtime_error("fixture logical support IDs/endpoints are invalid");
            endpoint_pairs.emplace_back(sources[ordinal], destinations[ordinal]);
            f.edge_sources[ordinal] = static_cast<std::uint64_t>(sources[ordinal]);
            ++expected_source_counts[f.edge_sources[ordinal]];
            const auto begin = f.base.participants.size();
            f.base.participants.push_back({0u, 0u,
                static_cast<std::uint64_t>(sources[ordinal]) + 1u});
            f.base.participants.push_back({1u, 0u,
                static_cast<std::uint64_t>(destinations[ordinal]) + 1u});
            f.base.edges.push_back({edge_ids[ordinal], begin, 2u, 0u});
            f.digest = mix_hash(f.digest, edge_ids[ordinal]);
            f.digest = mix_hash(f.digest, static_cast<std::uint64_t>(sources[ordinal]));
            f.digest = mix_hash(f.digest, static_cast<std::uint64_t>(destinations[ordinal]));
        }
        std::sort(endpoint_pairs.begin(), endpoint_pairs.end());
        if (std::adjacent_find(endpoint_pairs.begin(), endpoint_pairs.end())
            != endpoint_pairs.end())
            throw std::runtime_error("fixture contains duplicate source/destination pairs");
        for (const auto degree : expected_source_counts)
            if (degree > 512u)
                throw std::runtime_error("fixture source degree exceeds the declared 512 cap");
        std::vector<std::uint64_t> expected_source_offsets(cfg.entities + 1u, 0u);
        for (std::uint64_t entity = 0; entity < cfg.entities; ++entity)
            expected_source_offsets[entity + 1u] = expected_source_offsets[entity]
                + expected_source_counts[entity];
        std::vector<std::uint64_t> expected_source_ids(cfg.edges);
        auto source_cursor = expected_source_offsets;
        for (std::uint64_t ordinal = 0; ordinal < cfg.edges; ++ordinal)
            expected_source_ids[source_cursor[f.edge_sources[ordinal]]++] = edge_ids[ordinal];
        const auto source_indptr = read_raw_array<std::uint64_t>(directory,
            "source_edge_indptr", cfg.entities + 1u);
        const auto source_edge_ids = read_raw_u64_array(directory, "edge_ids");
        if (source_indptr != expected_source_offsets || source_edge_ids != expected_source_ids)
            throw std::runtime_error("fixture source CSR does not match support endpoints");
        f.base.view = {support::logical_support_schema_version_v1, 0u,
            7001u, 9001u, 1u, f.base.edges.data(), f.base.edges.size(),
            f.base.participants.data(), f.base.participants.size()};
        f.snapshot = {&f.base.view, nullptr, 1u};
        if (support::validate_support_snapshot_pair_v1(f.snapshot)
            != support::support_snapshot_status_v1::success)
            throw std::runtime_error("raw fixture support failed validation");
        f.gpu_edges = f.base.edges;
        f.gpu_participants = f.base.participants;
        f.query_offsets = read_raw_array<std::uint64_t>(directory,
            "cell_observed_predicate_offsets", cfg.cells + 1u);
        f.query_entities = read_raw_u64_array(directory, "cell_observed_predicate_ids");
        if (f.query_offsets.empty() || f.query_offsets.front() != 0u
            || f.query_offsets.back() != f.query_entities.size())
            throw std::runtime_error("fixture per-cell query CSR is malformed");
        for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
            if (f.query_offsets[cell] > f.query_offsets[cell + 1u]
                || f.query_offsets[cell + 1u] > f.query_entities.size())
                throw std::runtime_error("fixture per-cell query offsets are not monotone");
            for (auto q = f.query_offsets[cell]; q < f.query_offsets[cell + 1u]; ++q)
                if (f.query_entities[q] >= cfg.entities)
                    throw std::runtime_error("fixture query entity is outside the support domain");
        }
        return f;
    }
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
    std::vector<std::uint64_t> capacity_offsets;
    std::vector<std::uint64_t> row_capacities;
    std::vector<std::uint64_t> row_counts;
    std::vector<indexing::indexed_work_item_v1> items;
    std::uint64_t count = 0;
};

host_candidates make_host_candidates(const options &cfg,
                                     const std::vector<std::uint64_t> &row_capacities) {
    if (row_capacities.size() != cfg.cells)
        throw std::runtime_error("candidate row capacity shape mismatch");
    host_candidates result;
    result.offsets.resize(cfg.cells + 1u);
    result.capacity_offsets.resize(cfg.cells + 1u);
    result.row_capacities = row_capacities;
    result.row_counts.resize(cfg.cells);
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        result.capacity_offsets[cell + 1u] = result.capacity_offsets[cell] + row_capacities[cell];
    }
    result.items.resize(static_cast<std::size_t>(result.capacity_offsets.back()));
    return result;
}

std::uint64_t posting_degree(const indexing::reverse_posting_view_v1 &index,
                             const indexing::posting_term_v1 &term) {
    const auto key = std::pair<std::uint32_t, std::uint64_t>{term.role, term.entity_id};
    const auto *first = std::lower_bound(index.entries, index.entries + index.entry_count,
        key, [](const indexing::reverse_posting_record_v1 &record, const auto &value) {
            return record.role < value.first
                || (record.role == value.first && record.entity_id < value.second);
        });
    const auto *last = std::upper_bound(first, index.entries + index.entry_count, key,
        [](const auto &value, const indexing::reverse_posting_record_v1 &record) {
            return value.first < record.role
                || (value.first == record.role && value.second < record.entity_id);
        });
    return static_cast<std::uint64_t>(last - first);
}

std::vector<std::uint64_t> candidate_row_capacity_upper_bounds(
        const options &cfg, const indexing::reverse_posting_view_v1 &index,
        const std::vector<std::uint64_t> &query_offsets,
        const std::vector<indexing::posting_term_v1> &terms) {
    std::vector<std::uint64_t> capacities(cfg.cells);
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        std::uint64_t upper = 0u;
        for (auto q = query_offsets[cell]; q < query_offsets[cell + 1u]; ++q) {
            const auto degree = posting_degree(index, terms[q]);
            upper = std::min<std::uint64_t>(cfg.edges, upper + degree);
        }
        // Keep a non-null per-row output pointer even for an empty result.
        capacities[cell] = std::max<std::uint64_t>(1u, upper);
    }
    return capacities;
}

void compact_host_candidate_rows(host_candidates &out) {
    out.offsets[0] = 0u;
    for (std::size_t cell = 0; cell < out.row_counts.size(); ++cell) {
        if (out.row_counts[cell] > out.row_capacities[cell])
            throw std::runtime_error("candidate output exceeded its posting-degree upper bound");
        out.offsets[cell + 1u] = out.offsets[cell] + out.row_counts[cell];
    }
    std::vector<indexing::indexed_work_item_v1> compact(
        static_cast<std::size_t>(out.offsets.back()));
    for (std::size_t cell = 0; cell < out.row_counts.size(); ++cell) {
        if (out.row_counts[cell] != 0u)
            std::copy_n(out.items.data() + out.capacity_offsets[cell],
                        static_cast<std::size_t>(out.row_counts[cell]),
                        compact.data() + out.offsets[cell]);
    }
    out.items = std::move(compact);
    out.count = out.offsets.back();
}

struct inputs {
    std::vector<indexing::posting_term_v1> primal_queries;
    std::vector<indexing::posting_term_v1> derivative_queries;
    std::vector<std::uint64_t> query_offsets;
    std::vector<std::uint64_t> local_dirty_offsets;
    std::vector<std::uint64_t> local_dirty_edges;
    std::vector<std::uint64_t> latent_dirty_offsets;
    std::vector<std::uint64_t> latent_dirty_edges;
    std::vector<std::uint64_t> local_changed_genes;
    std::uint64_t latent_changed_module = 0u;
    std::vector<std::uint8_t> predicates;
    std::vector<std::uint8_t> local_changed_predicates;
    std::vector<std::uint8_t> latent_changed_predicates;
    std::vector<gating::condition_node_v1> nodes;
    std::vector<gating::edge_condition_binding_v1> bindings;
    gating::condition_program_view_v1 program{};
    std::uint32_t predicate_count = 0u;
    std::uint64_t local_dirty_total = 0u;
    std::uint64_t latent_dirty_total = 0u;
    std::uint64_t condition_pair_count = 0u;
    double observed_gate_density = 0;
    std::uint64_t digest = 1469598103934665603ull;
};

inputs make_inputs(const options &cfg, const fixture &f) {
    inputs in;
    in.digest = mix_hash(in.digest, cfg.edges);
    in.digest = mix_hash(in.digest, cfg.cells);
    in.digest = mix_hash(in.digest, cfg.entities);
    in.digest = mix_hash(in.digest, cfg.seed);
    std::uint64_t rng = cfg.seed ^ 0xd1b54a32d192ed03ull;
    if (!f.query_offsets.empty()) {
        in.query_offsets = f.query_offsets;
        in.primal_queries.reserve(f.query_entities.size());
        for (const auto entity : f.query_entities)
            in.primal_queries.push_back({0u, 0u, entity + 1u});
        in.derivative_queries = in.primal_queries;
    } else {
        in.primal_queries.reserve(cfg.cells);
        in.derivative_queries.reserve(cfg.cells);
        in.query_offsets.resize(cfg.cells + 1u);
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
    }
    using gating::condition_opcode_v1;
    if (f.biological_conditions) {
        const auto predicate_count = cfg.entities + f.module_count;
        if (predicate_count > std::numeric_limits<std::uint32_t>::max()
            || cfg.edges > std::numeric_limits<std::uint32_t>::max() / 3u)
            throw std::runtime_error("fixture condition program exceeds node-index limits");
        in.predicate_count = static_cast<std::uint32_t>(predicate_count);
        in.nodes.reserve(std::min(cfg.edges, cfg.entities) * 3u);
        in.bindings.reserve(cfg.edges);
        std::map<std::pair<std::uint64_t, std::uint32_t>, std::uint32_t> roots;
        for (std::uint64_t edge = 0; edge < cfg.edges; ++edge) {
            const auto key = std::pair<std::uint64_t, std::uint32_t>{
                f.edge_sources[edge], f.edge_modules[edge]};
            auto root = roots.find(key);
            if (root == roots.end()) {
                if (in.nodes.size() > std::numeric_limits<std::uint32_t>::max() - 3u)
                    throw std::runtime_error("fixture condition program exceeds node-index limits");
                const auto first_child = static_cast<std::uint32_t>(in.nodes.size());
                in.nodes.push_back({condition_opcode_v1::predicate, 1u, 0u,
                    0u, 0u, static_cast<std::uint32_t>(key.first), 0u});
                in.nodes.push_back({condition_opcode_v1::predicate, 1u, 0u,
                    0u, 0u, static_cast<std::uint32_t>(cfg.entities + key.second), 0u});
                in.nodes.push_back({condition_opcode_v1::all_of, 1u, 0u,
                    first_child, 2u, 0u, 0u});
                root = roots.emplace(key, first_child + 2u).first;
            }
            in.bindings.push_back({edge, root->second, 0u});
        }
        in.condition_pair_count = roots.size();
        in.predicates.assign(cfg.cells * predicate_count, 0u);
        if (f.observed_offsets.size() != cfg.cells + 1u
            || f.observed_offsets.front() != 0u
            || f.observed_offsets.back() != f.observed_ids.size()
            || f.observed_ids.size() != f.observed_values.size()
            || f.latent_values.size() != cfg.cells * f.module_count
            || f.local_change_cells.size() != f.local_change_genes.size()
            || f.local_change_genes.size() != f.local_change_values.size()
            || f.latent_change_groups.size() != f.latent_change_modules.size())
            throw std::runtime_error("fixture condition state arrays have inconsistent shapes");
        for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
            const auto begin = f.observed_offsets[cell];
            const auto end = f.observed_offsets[cell + 1u];
            if (begin > end || end > f.observed_ids.size())
                throw std::runtime_error("fixture observed-state offsets are malformed");
            for (auto i = begin; i < end; ++i) {
                const auto gene = f.observed_ids[i];
                if (gene >= cfg.entities || f.observed_values[i] > 1u)
                    throw std::runtime_error("fixture observed-state predicate is invalid");
                in.predicates[cell * predicate_count + gene] = f.observed_values[i];
            }
            for (std::uint64_t module = 0; module < f.module_count; ++module) {
                const auto value = f.latent_values[cell * f.module_count + module];
                if (value > 1u)
                    throw std::runtime_error("fixture latent predicate is not boolean");
                in.predicates[cell * predicate_count + cfg.entities + module] = value;
            }
        }
        in.local_changed_predicates = in.predicates;
        in.latent_changed_predicates = in.predicates;
        in.local_changed_genes.assign(cfg.cells, std::numeric_limits<std::uint64_t>::max());
        for (std::size_t i = 0; i < f.local_change_cells.size(); ++i) {
            const auto cell = f.local_change_cells[i];
            if (cell >= cfg.cells) throw std::runtime_error("local change cell is out of range");
            if (in.local_changed_genes[cell] != std::numeric_limits<std::uint64_t>::max())
                continue; // one sparse gene dependency per cell
            const auto gene = f.local_change_genes[i];
            const auto value = f.local_change_values[i];
            if (gene >= cfg.entities || value > 1u)
                throw std::runtime_error("local gene change is invalid");
            auto &state = in.local_changed_predicates[cell * predicate_count + gene];
            if (state == value)
                throw std::runtime_error("selected local gene change does not change its value");
            state = value;
            in.local_changed_genes[cell] = gene;
        }
        if (std::find(in.local_changed_genes.begin(), in.local_changed_genes.end(),
                      std::numeric_limits<std::uint64_t>::max())
            != in.local_changed_genes.end())
            throw std::runtime_error("fixture lacks one local gene change per cell");

        // Use one latent module from the fixture's correlated change groups and
        // flip that same shared dependency across the cell batch.
        const auto group0 = std::find(f.latent_change_groups.begin(),
                                      f.latent_change_groups.end(), 0u);
        if (group0 == f.latent_change_groups.end())
            throw std::runtime_error("fixture has no shared latent change group");
        const auto selected_change = static_cast<std::size_t>(
            group0 - f.latent_change_groups.begin());
        in.latent_changed_module = f.latent_change_modules[selected_change];
        if (in.latent_changed_module >= f.module_count)
            throw std::runtime_error("fixture latent change module is out of range");
        for (std::uint64_t cell = 0; cell < cfg.cells; ++cell)
            in.latent_changed_predicates[cell * predicate_count + cfg.entities
                + in.latent_changed_module] ^= 1u;

        in.local_dirty_offsets.reserve(cfg.cells + 1u);
        in.latent_dirty_offsets.reserve(cfg.cells + 1u);
        for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
            in.local_dirty_offsets.push_back(in.local_dirty_edges.size());
            in.latent_dirty_offsets.push_back(in.latent_dirty_edges.size());
            for (std::uint64_t edge = 0; edge < cfg.edges; ++edge) {
                if (f.edge_sources[edge] == in.local_changed_genes[cell])
                    in.local_dirty_edges.push_back(edge);
                if (f.edge_modules[edge] == in.latent_changed_module)
                    in.latent_dirty_edges.push_back(edge);
            }
        }
        in.local_dirty_offsets.push_back(in.local_dirty_edges.size());
        in.latent_dirty_offsets.push_back(in.latent_dirty_edges.size());
        in.local_dirty_total = in.local_dirty_edges.size();
        in.latent_dirty_total = in.latent_dirty_edges.size();
        in.digest = mix_hash(in.digest, in.latent_changed_module);
        for (const auto gene : in.local_changed_genes) in.digest = mix_hash(in.digest, gene);
        for (const auto ordinal : in.local_dirty_edges) in.digest = mix_hash(in.digest, ordinal);
        for (const auto ordinal : in.latent_dirty_edges) in.digest = mix_hash(in.digest, ordinal);
    } else {
        in.nodes = {
            {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 0u, 0u},
            {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 1u, 0u},
            {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 2u, 0u},
            {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 3u, 0u},
            {condition_opcode_v1::all_of, 1u, 0u, 0u, 2u, 0u, 0u},
            {condition_opcode_v1::any_of, 1u, 0u, 2u, 2u, 0u, 0u},
            {condition_opcode_v1::all_of, 1u, 0u, 0u, 3u, 0u, 0u},
        };
        const auto dirty_per_cell = cfg.dirty_fraction == 0.0 ? 0u
            : std::max<std::uint64_t>(1u, static_cast<std::uint64_t>(
                std::ceil(cfg.edges * cfg.dirty_fraction)));
        in.predicate_count = 4u;
        in.bindings.reserve(cfg.edges);
        for (std::uint64_t edge = 0; edge < cfg.edges; ++edge) {
            std::uint32_t root = 0u;
            if (edge < dirty_per_cell)
                root = edge % 3u == 0u ? 0u : edge % 3u == 1u ? 4u : 6u;
            else {
                const std::array<std::uint32_t, 4> roots{{1u, 2u, 3u, 5u}};
                root = roots[(edge - dirty_per_cell) % roots.size()];
            }
            in.bindings.push_back({edge, root, 0u});
            in.digest = mix_hash(in.digest, edge);
            in.digest = mix_hash(in.digest, root);
        }
        in.predicates.resize(cfg.cells * in.predicate_count);
        rng = cfg.seed ^ 0x94d049bb133111ebull;
        for (auto &predicate : in.predicates)
            predicate = unit_random(rng) < cfg.gate_density ? 1u : 0u;
        in.local_changed_predicates = in.predicates;
        in.latent_changed_predicates = in.predicates;
        for (std::uint64_t cell = 0; cell < cfg.cells; ++cell)
            in.local_changed_predicates[cell * in.predicate_count] ^= 1u;
        in.latent_changed_predicates = in.local_changed_predicates;
        in.local_dirty_offsets.reserve(cfg.cells + 1u);
        in.latent_dirty_offsets.reserve(cfg.cells + 1u);
        for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
            in.local_dirty_offsets.push_back(in.local_dirty_edges.size());
            in.latent_dirty_offsets.push_back(in.latent_dirty_edges.size());
            for (std::uint64_t ordinal = 0; ordinal < dirty_per_cell; ++ordinal) {
                in.local_dirty_edges.push_back(ordinal);
                in.latent_dirty_edges.push_back(ordinal);
            }
        }
        in.local_dirty_offsets.push_back(in.local_dirty_edges.size());
        in.latent_dirty_offsets.push_back(in.latent_dirty_edges.size());
        in.local_dirty_total = in.local_dirty_edges.size();
        in.latent_dirty_total = in.latent_dirty_edges.size();
    }
    in.program = {1u, in.nodes.data(), in.nodes.size(), in.bindings.data(),
        in.bindings.size(), cfg.edges, in.predicate_count, 0u};
    for (auto value : in.predicates) in.digest = mix_hash(in.digest, value);
    std::uint64_t gate_pass = 0u;
    if (f.biological_conditions) {
        for (std::uint64_t cell = 0; cell < cfg.cells; ++cell)
            for (std::uint64_t edge = 0; edge < cfg.edges; ++edge)
                gate_pass += in.predicates[cell * in.predicate_count + f.edge_sources[edge]]
                    != 0u && in.predicates[cell * in.predicate_count + cfg.entities
                        + f.edge_modules[edge]] != 0u;
    } else {
        for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
            const auto *p = in.predicates.data() + cell * in.predicate_count;
            for (const auto &binding : in.bindings) {
                const auto r = binding.root_node;
                const bool pass = r == 0u ? p[0] : r == 1u ? p[1] : r == 2u ? p[2]
                    : r == 3u ? p[3] : r == 4u ? p[0] && p[1]
                    : r == 5u ? p[2] || p[3] : p[0] && p[1] && p[2];
                gate_pass += pass;
            }
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
    return in;
}

void host_scan(const options &cfg, const fixture &f,
               const std::vector<std::uint64_t> &query_offsets,
               const std::vector<indexing::posting_term_v1> &terms,
               host_candidates &out) {
    out.count = 0u;
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        std::vector<std::uint8_t> role0(cfg.entities + 1u, 0u);
        std::vector<std::uint8_t> role1(cfg.entities + 1u, 0u);
        for (auto q = query_offsets[cell]; q < query_offsets[cell + 1u]; ++q) {
            const auto &term = terms[q];
            auto &membership = term.role == 0u ? role0 : role1;
            if (term.entity_id < membership.size()) membership[term.entity_id] = 1u;
        }
        std::uint64_t local_count = 0u;
        for (std::uint64_t edge = 0; edge < cfg.edges; ++edge) {
            bool match = false;
            const auto &item = f.gpu_edges[edge];
            for (std::uint32_t slot = 0; slot < item.participant_count; ++slot) {
                const auto &participant = f.gpu_participants[item.participant_begin + slot];
                if (participant.role == 0u && role0[participant.entity_id]) match = true;
                if (participant.role == 1u && role1[participant.entity_id]) match = true;
                if (match) break;
            }
            if (!match) continue;
            if (local_count >= out.row_capacities[cell])
                throw std::runtime_error("full scan exceeded its posting-degree output bound");
            out.items[out.capacity_offsets[cell] + local_count++] = {
                f.gpu_edges[edge].logical_edge_id, edge};
        }
        out.row_counts[cell] = local_count;
    }
    compact_host_candidate_rows(out);
}

void host_postings(const options &cfg,
                   const indexing::reverse_posting_view_v1 &index,
                   const std::vector<std::uint64_t> &query_offsets,
                   const std::vector<indexing::posting_term_v1> &terms,
                   std::vector<indexing::indexed_work_item_v1> &scratch,
                   host_candidates &out) {
    out.count = 0u;
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        indexing::candidate_output_v1 output{
            out.items.data() + out.capacity_offsets[cell],
            out.row_capacities[cell], 0u, 0u};
        const auto status = indexing::query_reverse_postings_v1(index,
            terms.data() + query_offsets[cell],
            query_offsets[cell + 1u] - query_offsets[cell],
            indexing::posting_match_v1::any,
            {scratch.data(), scratch.size()}, &output);
        if (status != indexing::posting_status_v1::success)
            throw std::runtime_error("host posting query failed");
        out.row_counts[cell] = output.written;
    }
    compact_host_candidate_rows(out);
}

struct host_gate {
    std::vector<std::uint64_t> bitmap;
    std::vector<std::uint8_t> node_scratch;
    std::vector<std::uint64_t> local_changed_bitmap;
    std::vector<std::uint64_t> latent_changed_bitmap;
    std::uint64_t words_per_cell = 0;
};

std::vector<std::uint64_t> direct_biological_gate_oracle(
        const options &cfg, const fixture &f, const inputs &in,
        const std::vector<std::uint8_t> &predicates) {
    const auto words_per_cell = cfg.edges / 64u + (cfg.edges % 64u != 0u);
    std::vector<std::uint64_t> words(cfg.cells * words_per_cell, 0u);
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        const auto *state = predicates.data() + cell * in.predicate_count;
        for (std::uint64_t edge = 0; edge < cfg.edges; ++edge) {
            const auto source = f.edge_sources[edge];
            const auto module = f.edge_modules[edge];
            if (state[source] != 0u && state[cfg.entities + module] != 0u)
                words[cell * words_per_cell + edge / 64u] |= std::uint64_t{1u} << (edge % 64u);
        }
    }
    return words;
}

host_gate make_host_gate(const options &cfg, const fixture &f, const inputs &in) {
    host_gate g;
    g.words_per_cell = cfg.edges / 64u + (cfg.edges % 64u != 0u);
    g.bitmap.resize(cfg.cells * g.words_per_cell);
    g.node_scratch.resize(in.nodes.size());
    const auto status = gating::evaluate_condition_bitmaps_v1(in.program, cfg.cells,
        in.predicates.data(), in.predicates.size(), g.bitmap.data(), g.bitmap.size(),
        g.node_scratch.data(), g.node_scratch.size());
    if (status != gating::condition_status_v1::success)
        throw std::runtime_error("host bitmap oracle failed");
    if (f.biological_conditions
        && g.bitmap != direct_biological_gate_oracle(cfg, f, in, in.predicates))
        throw std::runtime_error("condition DAG differs from direct biological gate oracle");

    auto validate_delta = [&](const std::vector<std::uint8_t> &changed_predicates,
                              const std::vector<std::uint64_t> &dirty_offsets,
                              const std::vector<std::uint64_t> &dirty_edges,
                              std::vector<std::uint64_t> &changed_bitmap,
                              const char *label) {
        changed_bitmap.resize(cfg.cells * g.words_per_cell);
        std::vector<std::uint8_t> scratch(in.nodes.size());
        const auto full_status = gating::evaluate_condition_bitmaps_v1(in.program, cfg.cells,
            changed_predicates.data(), changed_predicates.size(), changed_bitmap.data(),
            changed_bitmap.size(), scratch.data(), scratch.size());
        if (full_status != gating::condition_status_v1::success)
            throw std::runtime_error(std::string("host full changed bitmap failed: ") + label);
        if (f.biological_conditions
            && changed_bitmap != direct_biological_gate_oracle(cfg, f, in, changed_predicates))
            throw std::runtime_error(std::string("changed condition DAG differs from direct oracle: ")
                                     + label);

        auto incrementally_updated = g.bitmap;
        for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
            const auto begin = dirty_offsets[cell];
            const auto end = dirty_offsets[cell + 1u];
            const auto *dirty = begin == end ? nullptr : dirty_edges.data() + begin;
            const auto update_status = gating::update_condition_bitmap_v1(in.program,
                changed_predicates.data() + cell * in.predicate_count, in.predicate_count,
                dirty, end - begin, incrementally_updated.data() + cell * g.words_per_cell,
                g.words_per_cell, scratch.data(), scratch.size());
            if (update_status != gating::condition_status_v1::success)
                throw std::runtime_error(std::string("host dirty bitmap update failed: ") + label);
        }
        if (incrementally_updated != changed_bitmap)
            throw std::runtime_error(std::string("dirty edges do not cover exact changed gate set: ")
                                     + label);
    };
    validate_delta(in.local_changed_predicates, in.local_dirty_offsets,
        in.local_dirty_edges, g.local_changed_bitmap, "local-gene");
    validate_delta(in.latent_changed_predicates, in.latent_dirty_offsets,
        in.latent_dirty_edges, g.latent_changed_bitmap, "shared-latent");
    return g;
}

struct host_worksets {
    std::vector<std::uint64_t> primal_offsets;
    std::vector<std::uint64_t> derivative_offsets;
    std::vector<indexing::indexed_work_item_v1> primal_items;
    std::vector<indexing::indexed_work_item_v1> derivative_items;
    gating::gate_workset_pair_v1 pair{};
};

host_worksets make_host_worksets(const options &cfg,
                                 std::uint64_t primal_capacity,
                                 std::uint64_t derivative_capacity) {
    host_worksets w;
    w.primal_offsets.resize(cfg.cells + 1u);
    w.derivative_offsets.resize(cfg.cells + 1u);
    w.primal_items.resize(static_cast<std::size_t>(primal_capacity));
    w.derivative_items.resize(static_cast<std::size_t>(derivative_capacity));
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
    const auto fixture_load_start = host_clock::now();
    fixture f = make_fixture(cfg);
    const double fixture_load_ms = std::chrono::duration<double, std::milli>(
        host_clock::now() - fixture_load_start).count();
    // The snapshot stores borrowed pointers into the fixture object. Rebind
    // after returning it by value so no pointer can refer to a pre-move stack
    // address if NRVO is not used.
    f.base.view.edges = f.base.edges.data();
    f.base.view.edge_count = f.base.edges.size();
    f.base.view.participants = f.base.participants.data();
    f.base.view.participant_count = f.base.participants.size();
    if (!f.delta.edges.empty()) {
        f.delta.view.edges = f.delta.edges.data();
        f.delta.view.edge_count = f.delta.edges.size();
        f.delta.view.participants = f.delta.participants.data();
        f.delta.view.participant_count = f.delta.participants.size();
    }
    f.snapshot = {&f.base.view, f.delta.edges.empty() ? nullptr : &f.delta.view,
        f.base.view.structure_epoch};
    if (support::validate_support_snapshot_pair_v1(f.snapshot)
        != support::support_snapshot_status_v1::success)
        throw std::runtime_error("returned support fixture snapshot failed validation");
    const auto state_prepare_start = host_clock::now();
    inputs in = make_inputs(cfg, f);
    const double state_program_prepare_ms = std::chrono::duration<double, std::milli>(
        host_clock::now() - state_prepare_start).count();
    const auto requirements = indexing::query_reverse_posting_requirements_v1(f.snapshot);
    std::vector<indexing::reverse_posting_record_v1> posting_scratch(requirements.record_capacity);
    std::vector<indexing::reverse_posting_record_v1> host_entries(requirements.maximum_unique_entries);
    indexing::reverse_posting_view_v1 host_index{};
    auto build_host_index = [&] {
        const auto status = indexing::build_reverse_postings_v1(f.snapshot,
            posting_scratch.data(), posting_scratch.size(), host_entries.data(),
            host_entries.size(), &host_index);
        if (status != indexing::posting_status_v1::success)
            throw std::runtime_error("host reverse posting build failed (status="
                + std::to_string(static_cast<unsigned>(status)) + ", record_capacity="
                + std::to_string(posting_scratch.size()) + ", entry_capacity="
                + std::to_string(host_entries.size()) + ")");
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

    std::vector<std::uint32_t> host_root_node_by_edge(cfg.edges);
    std::uint64_t root_map_required = 0u;
    const auto root_map_prepare_start = host_clock::now();
    const auto root_map_status = gating::prepare_condition_root_map_v1(in.program,
        host_root_node_by_edge.data(), host_root_node_by_edge.size(), &root_map_required);
    const double root_map_prepare_ms = std::chrono::duration<double, std::milli>(
        host_clock::now() - root_map_prepare_start).count();
    if (root_map_status != gating::condition_status_v1::success
        || root_map_required != cfg.edges)
        throw std::runtime_error("condition root-map preparation failed");

    const auto oracle_start = host_clock::now();
    const auto primal_row_caps = candidate_row_capacity_upper_bounds(
        cfg, host_index, in.query_offsets, in.primal_queries);
    const auto derivative_row_caps = candidate_row_capacity_upper_bounds(
        cfg, host_index, in.query_offsets, in.derivative_queries);
    const auto primal_capacity = std::accumulate(primal_row_caps.begin(),
        primal_row_caps.end(), std::uint64_t{0});
    const auto derivative_capacity = std::accumulate(derivative_row_caps.begin(),
        derivative_row_caps.end(), std::uint64_t{0});
    host_candidates expected_p = make_host_candidates(cfg, primal_row_caps);
    host_candidates expected_d = make_host_candidates(cfg, derivative_row_caps);
    host_candidates expected_post_p = make_host_candidates(cfg, primal_row_caps);
    host_candidates expected_post_d = make_host_candidates(cfg, derivative_row_caps);
    host_scan(cfg, f, in.query_offsets, in.primal_queries, expected_p);
    host_scan(cfg, f, in.query_offsets, in.derivative_queries, expected_d);
    std::vector<indexing::indexed_work_item_v1> host_query_scratch(
        std::max<std::uint64_t>(1u, host_index.entry_count));
    host_postings(cfg, host_index, in.query_offsets, in.primal_queries,
                  host_query_scratch, expected_post_p);
    host_postings(cfg, host_index, in.query_offsets, in.derivative_queries,
                  host_query_scratch, expected_post_d);
    compare_candidates(expected_p, expected_post_p, "host scan/posting primal");
    compare_candidates(expected_d, expected_post_d, "host scan/posting derivative");
    if (!f.cell_query_candidate_counts.empty())
        for (std::uint64_t cell = 0; cell < cfg.cells; ++cell)
            if (expected_p.offsets[cell + 1u] - expected_p.offsets[cell]
                != f.cell_query_candidate_counts[cell])
                throw std::runtime_error("posting union differs from fixture candidate oracle");
    host_gate expected_gate = make_host_gate(cfg, f, in);
    const auto derivative_work_capacity = primal_capacity + derivative_capacity;
    host_worksets expected_scan_work = make_host_worksets(
        cfg, primal_capacity, derivative_work_capacity);
    host_worksets expected_post_work = make_host_worksets(
        cfg, primal_capacity, derivative_work_capacity);
    make_host_worksets_call(cfg, in, expected_gate, expected_p, expected_d,
                            expected_scan_work);
    make_host_worksets_call(cfg, in, expected_gate, expected_post_p, expected_post_d,
                            expected_post_work);
    const double host_oracle_ms = std::chrono::duration<double, std::milli>(
        host_clock::now() - oracle_start).count();

    if (cfg.host_oracle_only) {
        std::cout << std::setprecision(9)
            << "{\"schema\":\"cellerator.sparse_support.cuda_index_bench.host_oracle.v1\","
            << "\"case_id\":\"" << cfg.case_id << "\",\"correctness\":\"pass\","
            << "\"host_setup_ms\":{\"fixture_load_and_validation\":" << fixture_load_ms
            << ",\"state_and_condition_program_prepare\":" << state_program_prepare_ms
            << ",\"prepared_condition_root_map\":" << root_map_prepare_ms << "},"
            << "\"fixture\":{\"edges\":" << cfg.edges << ",\"cells\":" << cfg.cells
            << ",\"entities\":" << cfg.entities << ",\"fixture_dir\":\""
            << cfg.fixture_dir << "\",\"query_terms\":" << in.primal_queries.size()
            << ",\"condition_nodes\":" << in.nodes.size()
            << ",\"shared_source_module_conditions\":" << in.condition_pair_count << "},"
            << "\"counts\":{\"candidate_union\":" << expected_p.count
            << ",\"condition_eligible_primal\":" << expected_post_work.pair.primal.written
            << ",\"condition_eligible_derivative_union\":"
            << expected_post_work.pair.derivative.written
            << ",\"local_dirty_edges\":" << in.local_dirty_total
            << ",\"latent_dirty_edges\":" << in.latent_dirty_total << "},"
            << "\"host_oracle_ms\":" << host_oracle_ms << "}\n";
        return;
    }

    device_context device = make_device_context();
    const std::uint64_t n = cfg.edges * cfg.cells;
    const std::uint64_t bitmap_words_per_cell = cfg.edges / 64u + (cfg.edges % 64u != 0u);
    const std::uint64_t bitmap_count = cfg.cells * bitmap_words_per_cell;
    device_buffer<support::logical_support_edge_v1> d_edges(f.gpu_edges.size());
    device_buffer<support::logical_support_participant_v1> d_participants(f.gpu_participants.size());
    device_buffer<indexing::posting_term_v1> d_index_terms(projection_terms.size());
    device_buffer<std::uint64_t> d_index_offsets(projection_offsets.size());
    device_buffer<indexing::indexed_work_item_v1> d_index_items(projection_items.size());
    device_buffer<std::uint32_t> d_root_node_by_edge(host_root_node_by_edge.size());
    device_buffer<std::uint64_t> d_query_offsets(in.query_offsets.size());
    device_buffer<indexing::posting_term_v1> d_primal_queries(in.primal_queries.size());
    device_buffer<indexing::posting_term_v1> d_derivative_queries(in.derivative_queries.size());
    device_buffer<std::uint8_t> d_predicates(in.predicates.size());
    device_buffer<std::uint8_t> d_changed_predicates(in.predicates.size());
    device_buffer<gating::condition_node_v1> d_nodes(in.nodes.size());
    device_buffer<gating::edge_condition_binding_v1> d_bindings(in.bindings.size());
    device_buffer<std::uint64_t> d_dirty_offsets(cfg.cells + 1u);
    device_buffer<std::uint64_t> d_dirty_edges(
        std::max(in.local_dirty_edges.size(), in.latent_dirty_edges.size()));
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
    copy_to_device(d_root_node_by_edge, host_root_node_by_edge, device.stream);
    copy_to_device(d_query_offsets, in.query_offsets, device.stream);
    copy_to_device(d_primal_queries, in.primal_queries, device.stream);
    copy_to_device(d_derivative_queries, in.derivative_queries, device.stream);
    copy_to_device(d_predicates, in.predicates, device.stream);
    copy_to_device(d_changed_predicates, in.local_changed_predicates, device.stream);
    copy_to_device(d_nodes, std::vector<gating::condition_node_v1>(in.nodes.begin(), in.nodes.end()), device.stream);
    copy_to_device(d_bindings, in.bindings, device.stream);
    copy_to_device(d_dirty_offsets, in.local_dirty_offsets, device.stream);
    copy_to_device(d_dirty_edges, in.local_dirty_edges, device.stream);

    indexing::cuda_reverse_posting_view_v1 device_index{
        host_index.relation_id, host_index.structure_id, host_index.structure_epoch,
        d_index_terms.data, d_index_offsets.data, projection_req.term_count,
        d_index_items.data, projection_req.item_count};
    gating::condition_program_view_v1 device_program{in.program.structure_epoch,
        d_nodes.data, in.program.node_count, d_bindings.data,
        in.program.binding_count, in.program.edge_count, in.program.predicate_count, 0u};
    gating::cuda_condition_program_v1 device_prepared_program{
        device_program, d_root_node_by_edge.data, host_root_node_by_edge.size()};

    // Device posting workspaces and outputs.
    device_buffer<indexing::cuda_posting_scratch_v1::cursor_v1> d_p_cursors(in.primal_queries.size());
    device_buffer<indexing::cuda_posting_scratch_v1::cursor_v1> d_d_cursors(in.derivative_queries.size());
    device_buffer<std::uint64_t> d_p_counts(cfg.cells), d_d_counts(cfg.cells);
    device_buffer<indexing::cuda_posting_status_v1> d_p_post_status(1u), d_d_post_status(1u);
    device_buffer<std::uint64_t> d_p_post_offsets(cfg.cells + 1u), d_d_post_offsets(cfg.cells + 1u);
    device_buffer<indexing::indexed_work_item_v1> d_p_post_items(primal_capacity),
        d_d_post_items(derivative_capacity);
    indexing::cuda_posting_scratch_v1 p_post_scratch{
        d_p_cursors.data, in.primal_queries.size(), d_p_counts.data, d_p_post_status.data};
    indexing::cuda_posting_scratch_v1 d_post_scratch{
        d_d_cursors.data, in.derivative_queries.size(), d_d_counts.data, d_d_post_status.data};
    indexing::cuda_posting_output_v1 p_post_output{
        d_p_post_offsets.data, cfg.cells + 1u, d_p_post_items.data, primal_capacity};
    indexing::cuda_posting_output_v1 d_post_output{
        d_d_post_offsets.data, cfg.cells + 1u, d_d_post_items.data, derivative_capacity};

    // Full-scan baseline workspaces and CUB exclusive-scan scratch.
    device_buffer<std::uint64_t> d_p_flags(n), d_d_flags(n), d_p_prefix(n), d_d_prefix(n);
    device_buffer<std::uint64_t> d_p_scan_offsets(cfg.cells + 1u), d_d_scan_offsets(cfg.cells + 1u);
    device_buffer<indexing::indexed_work_item_v1> d_p_scan_items(primal_capacity),
        d_d_scan_items(derivative_capacity);
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
        CUDA_CHECK(gating::cuda_evaluate_condition_bitmaps_v1(device_prepared_program,
            cfg.cells, d_predicates.data, in.predicates.size(), d_bitmaps.data,
            bitmap_count, d_node_scratch.data, d_node_scratch.size,
            d_gate_status.data, device.stream));
    };

    // Reserve only the posting-degree upper bounds. Per-row stage offsets keep
    // compact outputs bounded without truncating any possible candidate.
    device_buffer<std::uint64_t> d_p_stage_offsets(cfg.cells + 1u), d_d_stage_offsets(cfg.cells + 1u);
    device_buffer<indexing::indexed_work_item_v1> d_p_stage_items(primal_capacity),
        d_d_stage_items(derivative_work_capacity);
    device_buffer<std::uint64_t> d_p_counts_work(cfg.cells), d_d_counts_work(cfg.cells);
    device_buffer<std::uint64_t> d_p_work_offsets(cfg.cells + 1u), d_d_work_offsets(cfg.cells + 1u);
    device_buffer<indexing::indexed_work_item_v1> d_p_work_items(primal_capacity),
        d_d_work_items(derivative_work_capacity);
    std::vector<std::uint64_t> p_stage_offsets(cfg.cells + 1u), d_stage_offsets(cfg.cells + 1u);
    for (std::uint64_t c = 0; c < cfg.cells; ++c) {
        p_stage_offsets[c + 1u] = p_stage_offsets[c] + primal_row_caps[c];
        d_stage_offsets[c + 1u] = d_stage_offsets[c]
            + primal_row_caps[c] + derivative_row_caps[c];
    }
    if (p_stage_offsets.back() != primal_capacity
        || d_stage_offsets.back() != derivative_work_capacity)
        throw std::runtime_error("per-cell workset stage capacity accounting mismatch");
    copy_to_device(d_p_stage_offsets, p_stage_offsets, device.stream);
    copy_to_device(d_d_stage_offsets, d_stage_offsets, device.stream);
    gating::cuda_workset_scratch_v1 work_scratch{};
    work_scratch.primal_offsets = d_p_stage_offsets.data;
    work_scratch.primal_items = d_p_stage_items.data;
    work_scratch.primal_item_capacity = primal_capacity;
    work_scratch.derivative_offsets = d_d_stage_offsets.data;
    work_scratch.derivative_items = d_d_stage_items.data;
    work_scratch.derivative_item_capacity = derivative_work_capacity;
    work_scratch.primal_counts = d_p_counts_work.data;
    work_scratch.derivative_counts = d_d_counts_work.data;
    work_scratch.status = d_gate_status.data;
    work_scratch.primal_offsets_capacity = cfg.cells + 1u;
    work_scratch.derivative_offsets_capacity = cfg.cells + 1u;
    gating::cuda_output_csr_v1 p_work_output{
        d_p_work_offsets.data, cfg.cells + 1u, d_p_work_items.data, primal_capacity};
    gating::cuda_output_csr_v1 d_work_output{
        d_d_work_offsets.data, cfg.cells + 1u, d_d_work_items.data, derivative_work_capacity};
    auto gate_scan_worksets = [&] {
        // item_count is the accessible buffer capacity here. The CSR final
        // offset delimits the populated prefix; this keeps the pipeline
        // device-resident without a host count synchronization.
        const gating::cuda_candidate_csr_v1 p{d_p_scan_offsets.data, d_p_scan_items.data,
            primal_capacity};
        const gating::cuda_candidate_csr_v1 d{d_d_scan_offsets.data, d_d_scan_items.data,
            derivative_capacity};
        CUDA_CHECK(gating::cuda_build_condition_worksets_v1(device_program, cfg.cells,
            d_bitmaps.data, bitmap_count, p, d, p_work_output, d_work_output,
            work_scratch, device.stream));
    };
    auto gate_post_worksets = [&] {
        const gating::cuda_candidate_csr_v1 p{d_p_post_offsets.data, d_p_post_items.data,
            primal_capacity};
        const gating::cuda_candidate_csr_v1 d{d_d_post_offsets.data, d_d_post_items.data,
            derivative_capacity};
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
    const auto condition_root_map_upload_stats = measure_cuda(device.stream, cfg.warmup,
        cfg.samples, 1u, [&] {
            copy_to_device(d_root_node_by_edge, host_root_node_by_edge, device.stream);
        });
    // Re-upload all non-projection fixture inputs before exact validation.
    copy_to_device(d_edges, f.gpu_edges, device.stream);
    copy_to_device(d_participants, f.gpu_participants, device.stream);
    copy_to_device(d_query_offsets, in.query_offsets, device.stream);
    copy_to_device(d_primal_queries, in.primal_queries, device.stream);
    copy_to_device(d_derivative_queries, in.derivative_queries, device.stream);
    copy_to_device(d_predicates, in.predicates, device.stream);
    copy_to_device(d_changed_predicates, in.local_changed_predicates, device.stream);
    copy_to_device(d_root_node_by_edge, host_root_node_by_edge, device.stream);
    copy_to_device(d_nodes, std::vector<gating::condition_node_v1>(in.nodes.begin(), in.nodes.end()), device.stream);
    copy_to_device(d_bindings, in.bindings, device.stream);
    copy_to_device(d_dirty_offsets, in.local_dirty_offsets, device.stream);
    copy_to_device(d_dirty_edges, in.local_dirty_edges, device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));

    // Device correctness for the exhaustive baseline and the exact index.
    full_scan_device();
    postings_device();
    evaluate_full_bitmap();
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    const auto scan_p_offsets = copy_from_device(d_p_scan_offsets.data, cfg.cells + 1u);
    const auto scan_d_offsets = copy_from_device(d_d_scan_offsets.data, cfg.cells + 1u);
    const auto post_p_offsets = copy_from_device(d_p_post_offsets.data, cfg.cells + 1u);
    const auto post_d_offsets = copy_from_device(d_d_post_offsets.data, cfg.cells + 1u);
    auto copy_candidates = [](std::vector<std::uint64_t> offsets,
                              std::vector<indexing::indexed_work_item_v1> items) {
        host_candidates result;
        result.count = offsets.back();
        result.offsets = std::move(offsets);
        result.items = std::move(items);
        return result;
    };
    auto device_scan_p = copy_candidates(scan_p_offsets,
        copy_from_device(d_p_scan_items.data, scan_p_offsets.back()));
    auto device_scan_d = copy_candidates(scan_d_offsets,
        copy_from_device(d_d_scan_items.data, scan_d_offsets.back()));
    auto device_post_p = copy_candidates(post_p_offsets,
        copy_from_device(d_p_post_items.data, post_p_offsets.back()));
    auto device_post_d = copy_candidates(post_d_offsets,
        copy_from_device(d_d_post_items.data, post_d_offsets.back()));
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
    CUDA_CHECK(gating::cuda_update_condition_bitmaps_v1(device_prepared_program, cfg.cells,
        d_changed_predicates.data, in.local_changed_predicates.size(), d_dirty_offsets.data,
        d_dirty_edges.data, in.local_dirty_edges.size(), d_dirty_bitmaps.data, bitmap_count,
        d_node_scratch.data, d_node_scratch.size, d_gate_status.data, device.stream));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    const auto dirty_bitmap = copy_from_device(d_dirty_bitmaps.data, bitmap_count);
    if (dirty_bitmap != expected_gate.local_changed_bitmap)
        throw std::runtime_error("CUDA local dirty bitmap differs from full host rebuild");

    copy_to_device(d_dirty_bitmaps, expected_gate.bitmap, device.stream);
    copy_to_device(d_changed_predicates, in.latent_changed_predicates, device.stream);
    copy_to_device(d_dirty_offsets, in.latent_dirty_offsets, device.stream);
    copy_to_device(d_dirty_edges, in.latent_dirty_edges, device.stream);
    CUDA_CHECK(gating::cuda_update_condition_bitmaps_v1(device_prepared_program, cfg.cells,
        d_changed_predicates.data, in.latent_changed_predicates.size(), d_dirty_offsets.data,
        d_dirty_edges.data, in.latent_dirty_edges.size(), d_dirty_bitmaps.data, bitmap_count,
        d_node_scratch.data, d_node_scratch.size, d_gate_status.data, device.stream));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    const auto latent_dirty_bitmap = copy_from_device(d_dirty_bitmaps.data, bitmap_count);
    if (latent_dirty_bitmap != expected_gate.latent_changed_bitmap)
        throw std::runtime_error("CUDA shared-latent dirty bitmap differs from full host rebuild");

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
    copy_to_device(d_changed_predicates, in.local_changed_predicates, device.stream);
    copy_to_device(d_dirty_offsets, in.local_dirty_offsets, device.stream);
    copy_to_device(d_dirty_edges, in.local_dirty_edges, device.stream);
    copy_to_device(d_dirty_bitmaps, expected_gate.bitmap, device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    const auto bitmap_local_dirty_stats = measure_cuda(device.stream, cfg.warmup,
        cfg.samples, iterations, [&] {
            CUDA_CHECK(gating::cuda_update_condition_bitmaps_v1(device_prepared_program, cfg.cells,
                d_changed_predicates.data, in.local_changed_predicates.size(), d_dirty_offsets.data,
                d_dirty_edges.data, in.local_dirty_edges.size(), d_dirty_bitmaps.data, bitmap_count,
                d_node_scratch.data, d_node_scratch.size, d_gate_status.data, device.stream));
        });
    copy_to_device(d_changed_predicates, in.latent_changed_predicates, device.stream);
    copy_to_device(d_dirty_offsets, in.latent_dirty_offsets, device.stream);
    copy_to_device(d_dirty_edges, in.latent_dirty_edges, device.stream);
    copy_to_device(d_dirty_bitmaps, expected_gate.bitmap, device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    const auto bitmap_latent_dirty_stats = measure_cuda(device.stream, cfg.warmup,
        cfg.samples, iterations, [&] {
            CUDA_CHECK(gating::cuda_update_condition_bitmaps_v1(device_prepared_program, cfg.cells,
                d_changed_predicates.data, in.latent_changed_predicates.size(), d_dirty_offsets.data,
                d_dirty_edges.data, in.latent_dirty_edges.size(), d_dirty_bitmaps.data, bitmap_count,
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
        std::uint64_t required_root_map_entries = 0u;
        if (gating::prepare_condition_root_map_v1(in.program,
                host_root_node_by_edge.data(), host_root_node_by_edge.size(),
                &required_root_map_entries)
                != gating::condition_status_v1::success
            || required_root_map_entries != cfg.edges)
            throw std::runtime_error("condition root-map setup failed");
        copy_to_device(d_index_terms, projection_terms, device.stream);
        copy_to_device(d_index_offsets, projection_offsets, device.stream);
        copy_to_device(d_index_items, projection_items, device.stream);
        copy_to_device(d_root_node_by_edge, host_root_node_by_edge, device.stream);
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
        (in.primal_queries.size() + in.derivative_queries.size())
            * sizeof(indexing::cuda_posting_scratch_v1::cursor_v1)
        + 2u * cfg.cells * sizeof(std::uint64_t)
        + 2u * sizeof(indexing::cuda_posting_status_v1)
        + 2u * (cfg.cells + 1u) * sizeof(std::uint64_t)
        + (primal_capacity + derivative_capacity)
            * sizeof(indexing::indexed_work_item_v1);
    const auto rescan_workspace_bytes =
        4u * n * sizeof(std::uint64_t)
        + 2u * (cfg.cells + 1u) * sizeof(std::uint64_t)
        + (primal_capacity + derivative_capacity)
            * sizeof(indexing::indexed_work_item_v1)
        + std::max(p_scan_bytes, d_scan_bytes);
    const auto workset_workspace_bytes =
        4u * (cfg.cells + 1u) * sizeof(std::uint64_t)
        + (4u * primal_capacity + 2u * derivative_capacity)
            * sizeof(indexing::indexed_work_item_v1)
        + 2u * cfg.cells * sizeof(std::uint64_t);
    const auto state_and_gate_bytes =
        (cfg.cells + 1u) * sizeof(std::uint64_t)
        + (in.primal_queries.size() + in.derivative_queries.size())
            * sizeof(indexing::posting_term_v1)
        + 2u * in.predicates.size() * sizeof(std::uint8_t)
        + in.nodes.size() * sizeof(gating::condition_node_v1)
        + in.bindings.size() * sizeof(gating::edge_condition_binding_v1)
        + (cfg.cells + 1u) * sizeof(std::uint64_t)
        + std::max(in.local_dirty_edges.size(), in.latent_dirty_edges.size())
            * sizeof(std::uint64_t)
        + 2u * bitmap_count * sizeof(std::uint64_t)
        + cfg.cells * in.nodes.size() * sizeof(std::uint8_t)
        + sizeof(gating::cuda_workset_status_v1);
    const auto total_device_bytes = total_requested_device_bytes();
    const auto accounted_device_bytes = structure_bytes + projection_bytes
        + posting_workspace_bytes + rescan_workspace_bytes
        + workset_workspace_bytes + state_and_gate_bytes
        + host_root_node_by_edge.size() * sizeof(std::uint32_t);
    if (accounted_device_bytes != total_device_bytes)
        throw std::runtime_error("device allocation accounting mismatch");

    timing_stats host_oracle_stats{};
    host_oracle_stats.samples_ms = {host_oracle_ms};
    host_oracle_stats.minimum = host_oracle_ms;
    host_oracle_stats.p10 = host_oracle_ms;
    host_oracle_stats.median = host_oracle_ms;
    host_oracle_stats.p90 = host_oracle_ms;
    host_oracle_stats.maximum = host_oracle_ms;

    std::uint64_t max_query_terms_per_cell = 0u;
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell)
        max_query_terms_per_cell = std::max(max_query_terms_per_cell,
            in.query_offsets[cell + 1u] - in.query_offsets[cell]);

    std::cout << std::setprecision(9)
        << "{\"schema\":\"cellerator.sparse_support.cuda_index_bench.v1\","
        << "\"scope\":\"cuda_device_resident_and_setup_separate\","
        << "\"host_setup_ms\":{\"fixture_load_and_validation\":" << fixture_load_ms
        << ",\"state_and_condition_program_prepare\":" << state_program_prepare_ms
        << ",\"prepared_condition_root_map\":" << root_map_prepare_ms << "},"
        << "\"case_id\":\"" << cfg.case_id << "\",\"correctness\":\"exact_host_oracle_pass\","
        << "\"fixture\":{\"edges\":" << cfg.edges << ",\"cells\":" << cfg.cells
        << ",\"entities\":" << cfg.entities << ",\"delta_edges\":" << f.delta.edges.size()
        << ",\"query_terms\":" << in.primal_queries.size()
        << ",\"max_query_terms_per_cell\":" << max_query_terms_per_cell
        << ",\"full_rescan_pair_scan_volume\":" << n
        << ",\"fixture_dir\":\"" << cfg.fixture_dir << "\""
        << ",\"participant_roles\":3,\"observed_gate_density\":" << in.observed_gate_density
        << ",\"source_skew\":" << cfg.source_skew << ",\"dirty_fraction\":" << cfg.dirty_fraction
        << ",\"local_dirty_edges_total\":" << in.local_dirty_total
        << ",\"local_dirty_edges_mean_per_cell\":"
        << static_cast<double>(in.local_dirty_total) / static_cast<double>(cfg.cells)
        << ",\"shared_latent_module_id\":" << in.latent_changed_module
        << ",\"shared_latent_dirty_edges_total\":" << in.latent_dirty_total
        << ",\"shared_latent_dirty_edges_mean_per_cell\":"
        << static_cast<double>(in.latent_dirty_total) / static_cast<double>(cfg.cells)
        << ",\"seed\":" << cfg.seed
        << ",\"digest\":\"0x" << std::hex << mix_hash(f.digest, in.digest) << std::dec << "\"},"
        << "\"counts\":{\"primal_candidates\":" << expected_p.count
        << ",\"derivative_candidates\":" << expected_d.count
        << ",\"primal_work_items\":" << expected_post_work.pair.primal.written
        << ",\"derivative_union_work_items\":" << expected_post_work.pair.derivative.written
        << ",\"derivative_only_work_items\":" << derivative_only
        << ",\"primal_output_capacity_upper_bound\":" << primal_capacity
        << ",\"derivative_output_capacity_upper_bound\":" << derivative_capacity
        << ",\"derivative_workset_capacity_upper_bound\":" << derivative_work_capacity
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
        << ",\"prepared_condition_root_map\":"
        << host_root_node_by_edge.size() * sizeof(std::uint32_t)
        << ",\"gate_workset_workspace_and_outputs\":" << workset_workspace_bytes
        << "},\"timings_ms\":{";
    print_stats("host_posting_index_build", host_index_stats, true);
    print_stats("host_device_projection_prepare", projection_stats, true);
    print_stats("host_exact_oracle_and_capacity_reservation", host_oracle_stats, true);
    print_stats("posting_and_root_map_setup_h2d_stream_complete", setup_stats, true);
    print_stats("projection_h2d_only", projection_upload_stats, true);
    print_stats("prepared_condition_root_map_h2d_only", condition_root_map_upload_stats, true);
    print_stats("cuda_full_rescan_pair_discovery", scan_discovery_stats, true);
    print_stats("cuda_reverse_posting_pair_discovery", posting_discovery_stats, true);
    print_stats("cuda_condition_bitmap_full_batch", bitmap_full_stats, true);
    print_stats("cuda_condition_bitmap_dirty_local_gene", bitmap_local_dirty_stats, true);
    print_stats("cuda_condition_bitmap_dirty_shared_latent", bitmap_latent_dirty_stats, true);
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
