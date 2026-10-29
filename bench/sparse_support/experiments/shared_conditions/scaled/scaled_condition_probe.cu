#include <Cellerator/compute/gating/cuda_worksets.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace cellerator::compute::gating;

namespace {

void cuda_check(cudaError_t error, const char *what) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
}

template <class T> std::vector<T> read_binary(const std::string &path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path);
    const auto bytes = stream.tellg();
    if (bytes < 0 || static_cast<std::uint64_t>(bytes) % sizeof(T) != 0u)
        throw std::runtime_error("invalid binary size: " + path);
    std::vector<T> result(static_cast<std::size_t>(bytes) / sizeof(T));
    stream.seekg(0);
    if (!result.empty())
        stream.read(reinterpret_cast<char *>(result.data()), bytes);
    if (!stream) throw std::runtime_error("short read: " + path);
    return result;
}

template <class T> struct DeviceArray {
    T *data = nullptr;
    std::uint64_t count = 0u;
    DeviceArray() = default;
    DeviceArray(const DeviceArray &) = delete;
    DeviceArray &operator=(const DeviceArray &) = delete;
    ~DeviceArray() { if (data != nullptr) cudaFree(data); }

    void allocate(std::uint64_t n) {
        count = n;
        if (n != 0u) cuda_check(cudaMalloc(reinterpret_cast<void **>(&data), n * sizeof(T)),
                                "cudaMalloc");
    }
    void upload(const std::vector<T> &host) {
        if (data == nullptr && !host.empty()) allocate(host.size());
        if (host.size() > count) throw std::runtime_error("device upload exceeds capacity");
        if (!host.empty()) cuda_check(cudaMemcpy(data, host.data(), host.size() * sizeof(T),
                                                 cudaMemcpyHostToDevice), "H2D copy");
    }
    void upload_one(const T &host) {
        if (data == nullptr) allocate(1u);
        cuda_check(cudaMemcpy(data, &host, sizeof(T), cudaMemcpyHostToDevice), "H2D scalar");
    }
    void reset_from(const DeviceArray<T> &source) {
        if (count != source.count) throw std::runtime_error("device reset capacity mismatch");
        if (count != 0u) cuda_check(cudaMemcpy(data, source.data, count * sizeof(T),
                                               cudaMemcpyDeviceToDevice), "D2D reset");
    }
};

template <class T> std::vector<T> download(const DeviceArray<T> &source) {
    std::vector<T> result(source.count);
    if (!result.empty()) cuda_check(cudaMemcpy(result.data(), source.data,
        result.size() * sizeof(T), cudaMemcpyDeviceToHost), "D2H copy");
    return result;
}

struct ChangeSet {
    std::vector<std::uint32_t> cell_offsets;
    std::vector<std::uint32_t> cells;
    std::vector<std::uint32_t> predicates;
    std::vector<std::uint8_t> values;
    std::uint64_t dirty_edge_count = 0u;
    std::uint64_t root_visit_count = 0u;
    std::uint64_t effective_state_change_count = 0u;
};

struct RootPair {
    std::uint32_t first = 0u;
    std::uint32_t second = 0u;
};

struct Fixture {
    std::string label;
    std::string directory;
    std::uint64_t edges = 0u;
    std::uint32_t cells = 0u;
    std::uint32_t genes = 0u;
    std::uint32_t modules = 0u;
    std::uint32_t predicates = 0u;
    std::uint64_t words_per_cell = 0u;
    std::vector<std::uint64_t> logical_edge_ids;
    std::vector<std::uint32_t> predicate_pair; // indexed by logical edge ordinal
    std::vector<std::uint8_t> initial_predicates; // row-major [cell,predicate]
    std::vector<std::uint64_t> expected_initial_bitmap;
    std::vector<std::uint64_t> initial_bitmap;
    std::vector<std::uint64_t> source_degree;
    std::vector<std::uint64_t> latent_degree;
    ChangeSet local_changes;
    ChangeSet latent_changes;
};

struct Compiled {
    std::vector<condition_node_v1> nodes;
    std::vector<edge_condition_binding_v1> bindings;
    std::vector<std::uint32_t> root_node_by_edge;
    std::vector<std::uint32_t> edge_root;
    std::vector<RootPair> roots;
    std::vector<std::uint64_t> root_edge_offsets;
    std::vector<std::uint32_t> root_edges;
    std::vector<std::uint64_t> predicate_root_offsets;
    std::vector<std::uint32_t> predicate_roots;
    std::vector<std::uint64_t> predicate_edge_offsets;
    std::vector<std::uint32_t> predicate_edges;
    std::vector<std::uint8_t> initial_root_values;
};

std::vector<std::uint32_t> make_cell_offsets(
    std::uint32_t cells, const std::vector<std::uint32_t> &cell_ids) {
    std::vector<std::uint32_t> offsets(cells + 1u, 0u);
    for (const auto cell : cell_ids) {
        if (cell >= cells) throw std::runtime_error("change cell ID out of range");
        ++offsets[cell + 1u];
    }
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    return offsets;
}

ChangeSet make_change_set(std::uint32_t cells, std::uint32_t genes,
                          std::uint32_t modules,
                          const std::vector<std::int32_t> &local_cells,
                          const std::vector<std::int64_t> &local_ids,
                          const std::vector<std::uint8_t> &local_values,
                          const std::vector<std::int32_t> &shared_cells,
                          const std::vector<std::int32_t> &shared_ids,
                          const std::vector<std::uint8_t> &shared_values,
                          bool latent) {
    struct Record { std::uint32_t cell, predicate; std::uint8_t value; std::uint64_t order; };
    std::vector<Record> records;
    if (latent) {
        if (shared_cells.size() != shared_ids.size()
            || shared_cells.size() != shared_values.size())
            throw std::runtime_error("latent change arrays have unequal lengths");
        records.reserve(shared_cells.size());
        for (std::size_t i = 0; i < shared_cells.size(); ++i) {
            const auto module = shared_ids[i];
            if (module < 0 || static_cast<std::uint32_t>(module) >= modules)
                throw std::runtime_error("shared-state change module ID out of range");
            records.push_back({static_cast<std::uint32_t>(shared_cells[i]),
                genes + static_cast<std::uint32_t>(module), shared_values[i], i});
        }
    } else {
        if (local_cells.size() != local_ids.size()
            || local_cells.size() != local_values.size())
            throw std::runtime_error("local change arrays have unequal lengths");
        records.reserve(local_cells.size());
        for (std::size_t i = 0; i < local_cells.size(); ++i) {
            const auto gene = local_ids[i];
            if (gene < 0 || static_cast<std::uint64_t>(gene) >= genes)
                throw std::runtime_error("local predicate change ID out of range");
            records.push_back({static_cast<std::uint32_t>(local_cells[i]),
                static_cast<std::uint32_t>(gene), local_values[i], i});
        }
    }
    std::stable_sort(records.begin(), records.end(), [](const auto &a, const auto &b) {
        return a.cell < b.cell;
    });
    ChangeSet changes;
    changes.cells.reserve(records.size());
    changes.predicates.reserve(records.size());
    changes.values.reserve(records.size());
    for (const auto &r : records) {
        changes.cells.push_back(r.cell);
        changes.predicates.push_back(r.predicate);
        changes.values.push_back(r.value);
    }
    changes.cell_offsets = make_cell_offsets(cells, changes.cells);
    for (std::uint32_t cell = 0; cell < cells; ++cell) {
        for (std::uint32_t i = changes.cell_offsets[cell];
             i < changes.cell_offsets[cell + 1u]; ++i) {
            for (std::uint32_t j = i + 1u; j < changes.cell_offsets[cell + 1u]; ++j)
                if (changes.predicates[i] == changes.predicates[j])
                    throw std::runtime_error("fixture repeats a changed predicate in one cell");
        }
    }
    return changes;
}

Fixture load_fixture(const std::string &label, const std::string &directory,
                     std::uint64_t edges, std::uint32_t cells,
                     std::uint32_t genes, std::uint32_t modules,
                     bool ranked_presence) {
    Fixture f;
    f.label = label;
    f.directory = directory;
    f.edges = edges;
    f.cells = cells;
    f.genes = genes;
    f.modules = modules;
    f.predicates = genes + modules;
    f.words_per_cell = edges / 64u + (edges % 64u != 0u ? 1u : 0u);
    const auto file = [&](const char *name) { return directory + "/" + name + ".bin"; };

    const auto edge_ids = read_binary<std::int64_t>(file("edge_ids"));
    const auto edge_ordinals = read_binary<std::int64_t>(file("edge_ordinal"));
    const auto condition_offsets = read_binary<std::int64_t>(file("edge_condition_indptr"));
    const auto condition_predicates = read_binary<std::int64_t>(file("edge_condition_predicate_id"));
    if (edge_ids.size() != edges || edge_ordinals.size() != edges
        || condition_offsets.size() != edges + 1u
        || condition_predicates.size() != 2u * edges)
        throw std::runtime_error("condition fixture arrays disagree with declared shape");

    f.logical_edge_ids.assign(edges, 0u);
    f.predicate_pair.assign(2u * edges, 0u);
    std::vector<std::uint8_t> ordinal_seen(edges, 0u);
    for (std::uint64_t row = 0; row < edges; ++row) {
        const auto ordinal = edge_ordinals[row];
        if (ordinal < 0 || static_cast<std::uint64_t>(ordinal) >= edges
            || ordinal_seen[ordinal])
            throw std::runtime_error("edge ordinal is not a permutation");
        ordinal_seen[ordinal] = 1u;
        if (condition_offsets[row + 1u] - condition_offsets[row] != 2)
            throw std::runtime_error("fixture edge is not a two-participant condition");
        const auto a = condition_predicates[condition_offsets[row]];
        const auto b = condition_predicates[condition_offsets[row] + 1u];
        if (a < 0 || b < 0 || static_cast<std::uint64_t>(a) >= f.predicates
            || static_cast<std::uint64_t>(b) >= f.predicates)
            throw std::runtime_error("condition participant ID is out of range");
        if (edge_ids[row] <= 0) throw std::runtime_error("logical edge ID must be positive");
        f.logical_edge_ids[ordinal] = static_cast<std::uint64_t>(edge_ids[row]);
        f.predicate_pair[2u * ordinal] = static_cast<std::uint32_t>(std::min(a, b));
        f.predicate_pair[2u * ordinal + 1u] = static_cast<std::uint32_t>(std::max(a, b));
    }

    const auto latent = read_binary<std::uint8_t>(file("gate_initial_state"));
    if (latent.size() != static_cast<std::uint64_t>(cells) * modules)
        throw std::runtime_error("latent state array disagrees with declared shape");
    f.initial_predicates.assign(static_cast<std::size_t>(cells) * f.predicates, 0u);
    const auto observed_offsets = read_binary<std::int64_t>(file(
        ranked_presence ? "rank_offsets" : "cell_observed_predicate_offsets"));
    const auto observed_ids = read_binary<std::int64_t>(file(
        ranked_presence ? "rank_entity_ids" : "cell_observed_predicate_ids"));
    const auto observed_values = read_binary<std::uint8_t>(file(
        ranked_presence ? "ranked_presence_values" : "cell_observed_predicate_values"));
    if (observed_offsets.size() != cells + 1u || observed_ids.size() != observed_values.size())
        throw std::runtime_error("local predicate state arrays disagree with declared shape");
    std::vector<std::int32_t> rank_positions;
    if (ranked_presence) {
        rank_positions = read_binary<std::int32_t>(file("rank_positions"));
        if (rank_positions.size() != observed_ids.size())
            throw std::runtime_error("rank position and ranked gene counts disagree");
    }
    for (std::uint32_t cell = 0; cell < cells; ++cell) {
        if (observed_offsets[cell] > observed_offsets[cell + 1u]
            || observed_offsets[cell + 1u] > observed_ids.size())
            throw std::runtime_error("observed predicate CSR offsets are invalid");
        if (ranked_presence && observed_offsets[cell + 1u] - observed_offsets[cell] != 512)
            throw std::runtime_error("ranked presence view must contain 512 genes per cell");
        std::vector<std::uint8_t> seen_gene(ranked_presence ? genes : 0u, 0u);
        std::array<std::uint8_t, 512> seen_rank{};
        for (std::int64_t i = observed_offsets[cell]; i < observed_offsets[cell + 1u]; ++i) {
            const auto id = observed_ids[i];
            if (id < 0 || static_cast<std::uint64_t>(id) >= genes || observed_values[i] > 1u)
                throw std::runtime_error("observed predicate value is invalid");
            if (ranked_presence && seen_gene[id]++)
                throw std::runtime_error("ranked presence view repeats a gene within a cell");
            if (ranked_presence) {
                const auto rank = rank_positions[i];
                if (rank < 0 || rank >= static_cast<std::int32_t>(seen_rank.size())
                    || seen_rank[rank]++)
                    throw std::runtime_error("ranked view positions are not a permutation");
            }
            f.initial_predicates[static_cast<std::size_t>(cell) * f.predicates + id]
                = observed_values[i];
        }
        for (std::uint32_t module = 0; module < modules; ++module) {
            const auto value = latent[static_cast<std::size_t>(cell) * modules + module];
            if (value > 1u) throw std::runtime_error("latent predicate value is not binary");
            f.initial_predicates[static_cast<std::size_t>(cell) * f.predicates
                                 + genes + module] = value;
        }
    }

    const auto local_cells = read_binary<std::int32_t>(file("local_gene_change_cell_id"));
    const auto local_ids = read_binary<std::int64_t>(file("local_gene_change_gene_id"));
    const auto local_values = read_binary<std::uint8_t>(file("local_gene_change_value"));
    const auto latent_cells = read_binary<std::int32_t>(file("gate_change_cell_id"));
    const auto latent_ids = read_binary<std::int32_t>(file("gate_change_predicate_id"));
    const auto latent_values = read_binary<std::uint8_t>(file("gate_change_value"));
    f.local_changes = make_change_set(cells, genes, modules, local_cells, local_ids,
        local_values, {}, {}, {}, false);
    f.latent_changes = make_change_set(cells, genes, modules, {}, {}, {}, latent_cells,
        latent_ids, latent_values, true);
    for (auto *changes : {&f.local_changes, &f.latent_changes})
        for (std::size_t i = 0; i < changes->cells.size(); ++i)
            changes->effective_state_change_count += static_cast<std::uint64_t>(
                f.initial_predicates[static_cast<std::size_t>(changes->cells[i])
                                     * f.predicates + changes->predicates[i]]
                != changes->values[i]);

    const auto local_fanout = read_binary<std::int64_t>(file("local_gene_change_edge_fanout"));
    const auto latent_fanout = read_binary<std::int64_t>(file("latent_change_edge_fanout"));
    if (local_fanout.size() != cells || latent_fanout.size() != cells)
        throw std::runtime_error("fixture fanout metadata has wrong cell count");
    f.source_degree.assign(f.predicates, 0u);
    f.latent_degree.assign(f.predicates, 0u);
    for (std::uint64_t edge = 0; edge < edges; ++edge) {
        ++f.source_degree[f.predicate_pair[2u * edge]];
        ++f.source_degree[f.predicate_pair[2u * edge + 1u]];
    }
    for (std::uint32_t cell = 0; cell < cells; ++cell) {
        std::uint64_t local_count = 0u, latent_count = 0u;
        for (std::uint32_t i = f.local_changes.cell_offsets[cell];
             i < f.local_changes.cell_offsets[cell + 1u]; ++i)
            local_count += f.source_degree[f.local_changes.predicates[i]];
        for (std::uint32_t i = f.latent_changes.cell_offsets[cell];
             i < f.latent_changes.cell_offsets[cell + 1u]; ++i)
            latent_count += f.source_degree[f.latent_changes.predicates[i]];
        f.local_changes.dirty_edge_count += local_count;
        f.latent_changes.dirty_edge_count += latent_count;
        if (local_count != static_cast<std::uint64_t>(local_fanout[cell])
            || latent_count != static_cast<std::uint64_t>(latent_fanout[cell]))
            throw std::runtime_error("condition participant fanout differs from frozen fixture");
    }

    return f;
}

Compiled compile_condition_geometry(const Fixture &f) {
    Compiled c;
    std::unordered_map<std::uint64_t, std::uint32_t> root_by_pair;
    root_by_pair.reserve(static_cast<std::size_t>(f.edges));
    c.edge_root.resize(f.edges);
    for (std::uint64_t edge = 0; edge < f.edges; ++edge) {
        const auto a = f.predicate_pair[2u * edge];
        const auto b = f.predicate_pair[2u * edge + 1u];
        const auto key = (static_cast<std::uint64_t>(a) << 32u) | b;
        auto found = root_by_pair.find(key);
        if (found == root_by_pair.end()) {
            if (c.roots.size() >= std::numeric_limits<std::uint32_t>::max() / 3u)
                throw std::runtime_error("condition root count exceeds node index format");
            const auto index = static_cast<std::uint32_t>(c.roots.size());
            c.roots.push_back({a, b});
            root_by_pair.emplace(key, index);
            c.edge_root[edge] = index;
        } else {
            c.edge_root[edge] = found->second;
        }
    }

    c.nodes.reserve(c.roots.size() * 3u);
    c.bindings.reserve(f.edges);
    c.root_node_by_edge.resize(f.edges);
    for (std::uint32_t root = 0; root < c.roots.size(); ++root) {
        const auto base = static_cast<std::uint32_t>(c.nodes.size());
        c.nodes.push_back({condition_opcode_v1::predicate, 1u, 0u, 0u, 0u,
                           c.roots[root].first, 0u});
        c.nodes.push_back({condition_opcode_v1::predicate, 1u, 0u, 0u, 0u,
                           c.roots[root].second, 0u});
        c.nodes.push_back({condition_opcode_v1::all_of, 1u, 0u, base, 2u, 0u, 0u});
    }
    for (std::uint64_t edge = 0; edge < f.edges; ++edge) {
        const auto root_node = 3u * c.edge_root[edge] + 2u;
        c.root_node_by_edge[edge] = root_node;
        c.bindings.push_back({edge, root_node, 0u});
    }

    std::vector<std::uint64_t> root_edge_counts(c.roots.size(), 0u);
    std::vector<std::uint64_t> predicate_root_counts(f.predicates, 0u);
    std::vector<std::uint64_t> predicate_edge_counts(f.predicates, 0u);
    for (const auto root : c.edge_root) ++root_edge_counts[root];
    for (const auto &root : c.roots) {
        ++predicate_root_counts[root.first];
        if (root.second != root.first) ++predicate_root_counts[root.second];
    }
    for (std::uint64_t edge = 0; edge < f.edges; ++edge) {
        ++predicate_edge_counts[f.predicate_pair[2u * edge]];
        if (f.predicate_pair[2u * edge + 1u] != f.predicate_pair[2u * edge])
            ++predicate_edge_counts[f.predicate_pair[2u * edge + 1u]];
    }
    c.root_edge_offsets.assign(c.roots.size() + 1u, 0u);
    std::partial_sum(root_edge_counts.begin(), root_edge_counts.end(),
                     c.root_edge_offsets.begin() + 1u);
    c.root_edges.resize(f.edges);
    auto root_edge_cursor = c.root_edge_offsets;
    for (std::uint64_t edge = 0; edge < f.edges; ++edge)
        c.root_edges[root_edge_cursor[c.edge_root[edge]]++] = static_cast<std::uint32_t>(edge);

    c.predicate_root_offsets.assign(static_cast<std::size_t>(f.predicates) + 1u, 0u);
    std::partial_sum(predicate_root_counts.begin(), predicate_root_counts.end(),
                     c.predicate_root_offsets.begin() + 1u);
    c.predicate_roots.resize(c.predicate_root_offsets.back());
    auto predicate_root_cursor = c.predicate_root_offsets;
    for (std::uint32_t root = 0; root < c.roots.size(); ++root) {
        c.predicate_roots[predicate_root_cursor[c.roots[root].first]++] = root;
        if (c.roots[root].second != c.roots[root].first)
            c.predicate_roots[predicate_root_cursor[c.roots[root].second]++] = root;
    }

    c.predicate_edge_offsets.assign(static_cast<std::size_t>(f.predicates) + 1u, 0u);
    std::partial_sum(predicate_edge_counts.begin(), predicate_edge_counts.end(),
                     c.predicate_edge_offsets.begin() + 1u);
    c.predicate_edges.resize(c.predicate_edge_offsets.back());
    auto predicate_edge_cursor = c.predicate_edge_offsets;
    for (std::uint64_t edge = 0; edge < f.edges; ++edge) {
        c.predicate_edges[predicate_edge_cursor[f.predicate_pair[2u * edge]]++]
            = static_cast<std::uint32_t>(edge);
        if (f.predicate_pair[2u * edge + 1u] != f.predicate_pair[2u * edge])
            c.predicate_edges[predicate_edge_cursor[f.predicate_pair[2u * edge + 1u]]++]
                = static_cast<std::uint32_t>(edge);
    }
    c.initial_root_values.resize(static_cast<std::size_t>(f.cells) * c.roots.size());
    for (std::uint32_t cell = 0; cell < f.cells; ++cell)
        for (std::uint32_t root = 0; root < c.roots.size(); ++root)
            c.initial_root_values[static_cast<std::size_t>(cell) * c.roots.size() + root]
                = static_cast<std::uint8_t>(
                    f.initial_predicates[static_cast<std::size_t>(cell) * f.predicates
                                         + c.roots[root].first]
                    && f.initial_predicates[static_cast<std::size_t>(cell) * f.predicates
                                            + c.roots[root].second]);
    return c;
}

std::vector<std::uint64_t> oracle_bitmap(const Fixture &f,
                                         const std::vector<std::uint8_t> &predicates) {
    std::vector<std::uint64_t> bitmap(static_cast<std::size_t>(f.cells)
                                      * f.words_per_cell, 0u);
    for (std::uint32_t cell = 0; cell < f.cells; ++cell) {
        const auto *state = predicates.data() + static_cast<std::size_t>(cell) * f.predicates;
        for (std::uint64_t edge = 0; edge < f.edges; ++edge) {
            const bool pass = state[f.predicate_pair[2u * edge]]
                           && state[f.predicate_pair[2u * edge + 1u]];
            if (pass)
                bitmap[static_cast<std::size_t>(cell) * f.words_per_cell + edge / 64u]
                    |= std::uint64_t{1} << (edge % 64u);
        }
    }
    return bitmap;
}

std::vector<std::uint8_t> apply_change_oracle(const Fixture &f,
    const ChangeSet &changes) {
    auto result = f.initial_predicates;
    for (std::size_t i = 0; i < changes.cells.size(); ++i)
        result[static_cast<std::size_t>(changes.cells[i]) * f.predicates
               + changes.predicates[i]] = changes.values[i];
    return result;
}

void require_bitmap_equal(const std::vector<std::uint64_t> &actual,
                          const std::vector<std::uint64_t> &expected,
                          const std::string &what) {
    if (actual.size() != expected.size()) throw std::runtime_error(what + " bitmap size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (actual[i] != expected[i]) {
            std::ostringstream message;
            message << what << " bitmap mismatch at word " << i << ": 0x"
                    << std::hex << actual[i] << " != 0x" << expected[i];
            throw std::runtime_error(message.str());
        }
}

__global__ void root_cache_full_kernel(
    const std::uint8_t *predicates, std::uint32_t cells, std::uint32_t predicate_count,
    const RootPair *roots, std::uint32_t root_count, std::uint8_t *root_values) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total = static_cast<std::uint64_t>(cells) * root_count;
    if (index >= total) return;
    const auto cell = static_cast<std::uint32_t>(index / root_count);
    const auto root = static_cast<std::uint32_t>(index % root_count);
    const auto *state = predicates + static_cast<std::uint64_t>(cell) * predicate_count;
    root_values[index] = static_cast<std::uint8_t>(state[roots[root].first]
                                                   && state[roots[root].second]);
}

__global__ void bitmap_from_root_cache_kernel(
    const std::uint32_t *edge_roots, std::uint64_t edge_count,
    std::uint32_t cells, std::uint32_t root_count, std::uint64_t words_per_cell,
    const std::uint8_t *root_values, std::uint64_t *bitmap) {
    const auto cell = static_cast<std::uint32_t>(blockIdx.x);
    const auto warp = threadIdx.x / 32u;
    const auto lane = threadIdx.x & 31u;
    const auto chunk = static_cast<std::uint64_t>(blockIdx.y) * 4u + warp;
    const auto chunks_per_cell = words_per_cell * 2u;
    if (cell >= cells || chunk >= chunks_per_cell) return;
    const auto edge = chunk * 32u + lane;
    bool pass = false;
    if (edge < edge_count)
        pass = root_values[static_cast<std::uint64_t>(cell) * root_count
                           + edge_roots[edge]] != 0u;
    const auto mask = __ballot_sync(0xffffffffu, pass);
    if (lane == 0u)
        reinterpret_cast<std::uint32_t *>(bitmap)[static_cast<std::uint64_t>(cell)
                                                   * chunks_per_cell + chunk] = mask;
}

__global__ void apply_predicate_changes_kernel(
    std::uint64_t change_count, const std::uint32_t *cells,
    const std::uint32_t *predicates, const std::uint8_t *values,
    std::uint32_t predicate_count, std::uint8_t *state) {
    const auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= change_count) return;
    state[static_cast<std::uint64_t>(cells[i]) * predicate_count + predicates[i]] = values[i];
}

__global__ void prepare_dirty_worklist_kernel(
    std::uint32_t cells, const std::uint32_t *change_cell_offsets,
    const std::uint32_t *change_predicates,
    const std::uint64_t *predicate_edge_offsets,
    std::uint64_t *dirty_offsets, std::uint64_t *change_output_offsets,
    std::uint64_t *total_items) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    std::uint64_t total = 0u;
    dirty_offsets[0] = 0u;
    for (std::uint32_t cell = 0; cell < cells; ++cell) {
        for (std::uint32_t i = change_cell_offsets[cell];
             i < change_cell_offsets[cell + 1u]; ++i) {
            const auto predicate = change_predicates[i];
            change_output_offsets[i] = total;
            total += predicate_edge_offsets[predicate + 1u]
                   - predicate_edge_offsets[predicate];
        }
        dirty_offsets[cell + 1u] = total;
    }
    *total_items = total;
}

__global__ void scatter_dirty_worklist_kernel(
    std::uint32_t change_count, const std::uint32_t *change_predicates,
    const std::uint64_t *predicate_edge_offsets,
    const std::uint32_t *predicate_edges,
    const std::uint64_t *change_output_offsets,
    std::uint64_t *dirty_edges) {
    const auto change = static_cast<std::uint32_t>(blockIdx.x);
    if (change >= change_count) return;
    const auto predicate = change_predicates[change];
    const auto begin = predicate_edge_offsets[predicate];
    const auto end = predicate_edge_offsets[predicate + 1u];
    const auto out = change_output_offsets[change];
    for (auto i = begin + threadIdx.x; i < end; i += blockDim.x)
        dirty_edges[out + i - begin] = predicate_edges[i];
}

__global__ void update_changed_root_cache_kernel(
    std::uint32_t change_count, const std::uint32_t *change_cells,
    const std::uint32_t *change_predicates,
    const std::uint32_t *change_cell_offsets,
    const std::uint64_t *predicate_root_offsets,
    const std::uint32_t *predicate_roots,
    const std::uint64_t *root_edge_offsets,
    const std::uint32_t *root_edges,
    const RootPair *roots, std::uint32_t root_count,
    std::uint32_t predicate_count, const std::uint8_t *state,
    std::uint8_t *root_values, std::uint64_t words_per_cell,
    std::uint64_t *bitmap) {
    const auto change = static_cast<std::uint32_t>(blockIdx.x);
    if (change >= change_count) return;
    const auto cell = change_cells[change];
    const auto predicate = change_predicates[change];
    const auto warp = threadIdx.x / 32u;
    const auto lane = threadIdx.x & 31u;
    const auto root_begin = predicate_root_offsets[predicate];
    const auto root_end = predicate_root_offsets[predicate + 1u];
    const auto cell_change_begin = change_cell_offsets[cell];
    const auto *cell_state = state + static_cast<std::uint64_t>(cell) * predicate_count;
    for (auto rpos = root_begin + warp; rpos < root_end; rpos += 4u) {
        const auto root = predicate_roots[rpos];
        std::uint32_t first_owner = 1u;
        if (lane == 0u) {
            for (auto prior = cell_change_begin; prior < change; ++prior) {
                const auto prior_predicate = change_predicates[prior];
                if (prior_predicate == roots[root].first
                    || prior_predicate == roots[root].second) {
                    first_owner = 0u;
                    break;
                }
            }
        }
        first_owner = __shfl_sync(0xffffffffu, first_owner, 0);
        if (first_owner == 0u) continue;
        std::uint32_t value = 0u;
        if (lane == 0u)
            value = static_cast<std::uint32_t>(cell_state[roots[root].first]
                                               && cell_state[roots[root].second]);
        value = __shfl_sync(0xffffffffu, value, 0);
        if (lane == 0u)
            root_values[static_cast<std::uint64_t>(cell) * root_count + root]
                = static_cast<std::uint8_t>(value);
        const auto edge_begin = root_edge_offsets[root];
        const auto edge_end = root_edge_offsets[root + 1u];
        for (auto i = edge_begin + lane; i < edge_end; i += 32u) {
            const auto edge = root_edges[i];
            auto *word = reinterpret_cast<unsigned long long *>(
                bitmap + static_cast<std::uint64_t>(cell) * words_per_cell + edge / 64u);
            const auto bit = static_cast<unsigned long long>(std::uint64_t{1} << (edge % 64u));
            if (value != 0u) atomicOr(word, bit);
            else atomicAnd(word, ~bit);
        }
    }
}

struct Timing { double device_ms = 0.0; double host_ms = 0.0; };

struct TimingStats {
    std::vector<double> device_ms;
    std::vector<double> host_ms;
};

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(p * static_cast<double>(values.size() - 1u));
    return values[index];
}

Timing summarize(const TimingStats &stats, const std::string &name,
                 const std::string &path, const std::string &stage,
                 const std::string &case_name) {
    const auto med_device = percentile(stats.device_ms, 0.5);
    const auto med_host = percentile(stats.host_ms, 0.5);
    std::cout << std::fixed << std::setprecision(6)
              << "{\"record\":\"timing\",\"case\":\"" << case_name
              << "\",\"stage\":\"" << stage << "\",\"path\":\"" << path
              << "\",\"metric\":\"cuda_event_ms\",\"median\":" << med_device
              << ",\"p10\":" << percentile(stats.device_ms, 0.1)
              << ",\"p90\":" << percentile(stats.device_ms, 0.9)
              << ",\"host_wall_median_ms\":" << med_host
              << ",\"host_wall_p10_ms\":" << percentile(stats.host_ms, 0.1)
              << ",\"host_wall_p90_ms\":" << percentile(stats.host_ms, 0.9)
              << ",\"samples\":" << stats.device_ms.size() << "}\n";
    return {med_device, med_host};
}

class Runner {
public:
    Fixture f;
    Compiled c;
    condition_program_view_v1 host_program{};
    condition_program_view_v1 device_program{};
    cuda_condition_program_v1 prepared_program{};
    DeviceArray<condition_node_v1> d_nodes;
    DeviceArray<edge_condition_binding_v1> d_bindings;
    DeviceArray<std::uint32_t> d_root_nodes;
    DeviceArray<std::uint32_t> d_edge_roots;
    DeviceArray<RootPair> d_roots;
    DeviceArray<std::uint64_t> d_root_edge_offsets;
    DeviceArray<std::uint32_t> d_root_edges;
    DeviceArray<std::uint64_t> d_pred_root_offsets;
    DeviceArray<std::uint32_t> d_pred_roots;
    DeviceArray<std::uint64_t> d_pred_edge_offsets;
    DeviceArray<std::uint32_t> d_pred_edges;
    DeviceArray<std::uint8_t> d_predicate_initial;
    DeviceArray<std::uint8_t> d_predicate_work;
    DeviceArray<std::uint8_t> d_initial_root_values;
    DeviceArray<std::uint8_t> d_root_values;
    DeviceArray<std::uint64_t> d_initial_bitmap;
    DeviceArray<std::uint64_t> d_bitmap;
    DeviceArray<std::uint8_t> d_node_scratch;
    DeviceArray<cuda_workset_status_v1> d_status;
    DeviceArray<std::uint32_t> d_local_cells, d_local_predicates;
    DeviceArray<std::uint8_t> d_local_values;
    DeviceArray<std::uint32_t> d_local_cell_offsets;
    DeviceArray<std::uint32_t> d_latent_cells, d_latent_predicates;
    DeviceArray<std::uint8_t> d_latent_values;
    DeviceArray<std::uint32_t> d_latent_cell_offsets;
    DeviceArray<std::uint64_t> d_dirty_offsets, d_dirty_change_offsets,
                               d_dirty_edges, d_dirty_total;
    cudaEvent_t event_begin = nullptr, event_end = nullptr;
    std::uint32_t samples = 11u;
    std::uint32_t warmups = 2u;

    Runner(Fixture fixture, Compiled compiled, std::uint32_t sample_count,
           std::uint32_t warmup_count)
        : f(std::move(fixture)), c(std::move(compiled)), samples(sample_count),
          warmups(warmup_count) {
        host_program = {1u, c.nodes.data(), c.nodes.size(), c.bindings.data(),
                        c.bindings.size(), f.edges, f.predicates, 0u};
        if (validate_condition_program_v1(host_program) != condition_status_v1::success)
            throw std::runtime_error("compiled condition program failed host validation");
        std::uint64_t required = 0u;
        c.root_node_by_edge.resize(f.edges);
        if (prepare_condition_root_map_v1(host_program, c.root_node_by_edge.data(),
                c.root_node_by_edge.size(), &required) != condition_status_v1::success
            || required != f.edges)
            throw std::runtime_error("could not prepare condition root map");
        f.expected_initial_bitmap = oracle_bitmap(f, f.initial_predicates);
        f.initial_bitmap = f.expected_initial_bitmap;
        upload_all();
        device_program = {1u, d_nodes.data, c.nodes.size(), d_bindings.data,
                          c.bindings.size(), f.edges, f.predicates, 0u};
        prepared_program = {device_program, d_root_nodes.data, f.edges};
        cuda_check(cudaEventCreate(&event_begin), "create start event");
        cuda_check(cudaEventCreate(&event_end), "create stop event");
    }

    ~Runner() {
        if (event_begin) cudaEventDestroy(event_begin);
        if (event_end) cudaEventDestroy(event_end);
    }

    void upload_all() {
        d_nodes.upload(c.nodes);
        d_bindings.upload(c.bindings);
        d_root_nodes.upload(c.root_node_by_edge);
        d_edge_roots.upload(c.edge_root);
        d_roots.upload(c.roots);
        d_root_edge_offsets.upload(c.root_edge_offsets);
        d_root_edges.upload(c.root_edges);
        d_pred_root_offsets.upload(c.predicate_root_offsets);
        d_pred_roots.upload(c.predicate_roots);
        d_pred_edge_offsets.upload(c.predicate_edge_offsets);
        d_pred_edges.upload(c.predicate_edges);
        d_predicate_initial.upload(f.initial_predicates);
        d_predicate_work.allocate(f.initial_predicates.size());
        d_initial_root_values.upload(c.initial_root_values);
        d_root_values.allocate(c.initial_root_values.size());
        d_initial_bitmap.upload(f.initial_bitmap);
        d_bitmap.allocate(f.initial_bitmap.size());
        d_node_scratch.allocate(static_cast<std::uint64_t>(f.cells) * c.nodes.size());
        d_status.allocate(1u);
        upload_changes(f.local_changes, d_local_cells, d_local_predicates,
                       d_local_values, d_local_cell_offsets);
        upload_changes(f.latent_changes, d_latent_cells, d_latent_predicates,
                       d_latent_values, d_latent_cell_offsets);

        const auto max_dirty = std::max(f.local_changes.dirty_edge_count,
                                        f.latent_changes.dirty_edge_count);
        d_dirty_offsets.allocate(static_cast<std::uint64_t>(f.cells) + 1u);
        d_dirty_change_offsets.allocate(std::max(f.local_changes.cells.size(),
                                                f.latent_changes.cells.size()));
        d_dirty_edges.allocate(max_dirty);
        d_dirty_total.allocate(1u);
    }

    static void upload_changes(const ChangeSet &changes,
        DeviceArray<std::uint32_t> &d_cells, DeviceArray<std::uint32_t> &d_predicates,
        DeviceArray<std::uint8_t> &d_values,
        DeviceArray<std::uint32_t> &d_cell_offsets) {
        d_cells.upload(changes.cells);
        d_predicates.upload(changes.predicates);
        d_values.upload(changes.values);
        d_cell_offsets.upload(changes.cell_offsets);
    }

    Timing measure(const std::function<void()> &fn) {
        cuda_check(cudaDeviceSynchronize(), "pre-timing synchronize");
        const auto start = std::chrono::steady_clock::now();
        cuda_check(cudaEventRecord(event_begin, nullptr), "record begin");
        fn();
        cuda_check(cudaEventRecord(event_end, nullptr), "record end");
        cuda_check(cudaEventSynchronize(event_end), "wait end event");
        const auto stop = std::chrono::steady_clock::now();
        float elapsed = 0.0f;
        cuda_check(cudaEventElapsedTime(&elapsed, event_begin, event_end), "event elapsed");
        return {elapsed, std::chrono::duration<double, std::milli>(stop - start).count()};
    }

    void run_native_full(bool prepared) {
        const auto error = prepared
            ? cuda_evaluate_condition_bitmaps_v1(prepared_program, f.cells,
                d_predicate_initial.data, d_predicate_initial.count, d_bitmap.data,
                d_bitmap.count, d_node_scratch.data, d_node_scratch.count,
                d_status.data, nullptr)
            : cuda_evaluate_condition_bitmaps_v1(device_program, f.cells,
                d_predicate_initial.data, d_predicate_initial.count, d_bitmap.data,
                d_bitmap.count, d_node_scratch.data, d_node_scratch.count,
                d_status.data, nullptr);
        cuda_check(error, "native full condition bitmap");
    }

    void run_candidate_full() {
        const auto total = static_cast<std::uint64_t>(f.cells) * c.roots.size();
        root_cache_full_kernel<<<static_cast<unsigned>((total + 255u) / 256u), 256>>>(
            d_predicate_initial.data, f.cells, f.predicates, d_roots.data,
            static_cast<std::uint32_t>(c.roots.size()), d_root_values.data);
        cuda_check(cudaGetLastError(), "candidate root cache full kernel");
        const auto chunks = d_bitmap.count / f.cells * 2u;
        bitmap_from_root_cache_kernel<<<dim3(f.cells, static_cast<unsigned>((chunks + 3u) / 4u)), 128>>>(
            d_edge_roots.data, f.edges, f.cells, c.roots.size(), f.words_per_cell,
            d_root_values.data, d_bitmap.data);
        cuda_check(cudaGetLastError(), "candidate bitmap materialization kernel");
    }

    void check_native_status(const std::string &path) {
        cuda_workset_status_v1 status{};
        cuda_check(cudaMemcpy(&status, d_status.data, sizeof(status),
                              cudaMemcpyDeviceToHost), "copy CUDA workset status");
        if (status.code != 0u)
            throw std::runtime_error(path + " returned CUDA workset status "
                                     + std::to_string(status.code));
    }

    std::vector<std::uint64_t> copy_bitmap() const { return download(d_bitmap); }

    void apply_changes(const ChangeSet &changes,
        const DeviceArray<std::uint32_t> &cells,
        const DeviceArray<std::uint32_t> &predicates,
        const DeviceArray<std::uint8_t> &values) {
        if (changes.cells.empty()) return;
        apply_predicate_changes_kernel<<<static_cast<unsigned>((changes.cells.size() + 255u) / 256u), 256>>>(
            changes.cells.size(), cells.data, predicates.data, values.data,
            f.predicates, d_predicate_work.data);
        cuda_check(cudaGetLastError(), "apply condition predicate changes");
    }

    void build_dirty(const ChangeSet &changes,
        const DeviceArray<std::uint32_t> &predicates,
        const DeviceArray<std::uint32_t> &cell_offsets) {
        prepare_dirty_worklist_kernel<<<1, 1>>>(f.cells, cell_offsets.data,
            predicates.data, d_pred_edge_offsets.data, d_dirty_offsets.data,
            d_dirty_change_offsets.data, d_dirty_total.data);
        cuda_check(cudaGetLastError(), "prepare dirty condition worklist");
        if (!changes.cells.empty()) {
            scatter_dirty_worklist_kernel<<<static_cast<unsigned>(changes.cells.size()), 128>>>(
                changes.cells.size(), predicates.data, d_pred_edge_offsets.data,
                d_pred_edges.data, d_dirty_change_offsets.data, d_dirty_edges.data);
            cuda_check(cudaGetLastError(), "scatter dirty condition worklist");
        }
    }

    void run_native_dirty(const ChangeSet &changes,
        const DeviceArray<std::uint32_t> &cells,
        const DeviceArray<std::uint32_t> &predicates,
        const DeviceArray<std::uint8_t> &values,
        const DeviceArray<std::uint32_t> &cell_offsets, bool prepared) {
        apply_changes(changes, cells, predicates, values);
        build_dirty(changes, predicates, cell_offsets);
        const auto error = prepared
            ? cuda_update_condition_bitmaps_v1(prepared_program, f.cells,
                d_predicate_work.data, d_predicate_work.count, d_dirty_offsets.data,
                d_dirty_edges.data, changes.dirty_edge_count, d_bitmap.data,
                d_bitmap.count, d_node_scratch.data, d_node_scratch.count,
                d_status.data, nullptr)
            : cuda_update_condition_bitmaps_v1(device_program, f.cells,
                d_predicate_work.data, d_predicate_work.count, d_dirty_offsets.data,
                d_dirty_edges.data, changes.dirty_edge_count, d_bitmap.data,
                d_bitmap.count, d_node_scratch.data, d_node_scratch.count,
                d_status.data, nullptr);
        cuda_check(error, "native dirty condition bitmap");
    }

    void run_native_full_update(const ChangeSet &changes,
        const DeviceArray<std::uint32_t> &cells,
        const DeviceArray<std::uint32_t> &predicates,
        const DeviceArray<std::uint8_t> &values, bool prepared) {
        apply_changes(changes, cells, predicates, values);
        const auto error = prepared
            ? cuda_evaluate_condition_bitmaps_v1(prepared_program, f.cells,
                d_predicate_work.data, d_predicate_work.count, d_bitmap.data,
                d_bitmap.count, d_node_scratch.data, d_node_scratch.count,
                d_status.data, nullptr)
            : cuda_evaluate_condition_bitmaps_v1(device_program, f.cells,
                d_predicate_work.data, d_predicate_work.count, d_bitmap.data,
                d_bitmap.count, d_node_scratch.data, d_node_scratch.count,
                d_status.data, nullptr);
        cuda_check(error, "native full rebuild after state update");
    }

    void run_candidate_full_update(const ChangeSet &changes,
        const DeviceArray<std::uint32_t> &cells,
        const DeviceArray<std::uint32_t> &predicates,
        const DeviceArray<std::uint8_t> &values) {
        apply_changes(changes, cells, predicates, values);
        const auto total = static_cast<std::uint64_t>(f.cells) * c.roots.size();
        root_cache_full_kernel<<<static_cast<unsigned>((total + 255u) / 256u), 256>>>(
            d_predicate_work.data, f.cells, f.predicates, d_roots.data,
            static_cast<std::uint32_t>(c.roots.size()), d_root_values.data);
        cuda_check(cudaGetLastError(), "candidate root cache full update");
        const auto chunks = d_bitmap.count / f.cells * 2u;
        bitmap_from_root_cache_kernel<<<dim3(f.cells, static_cast<unsigned>((chunks + 3u) / 4u)), 128>>>(
            d_edge_roots.data, f.edges, f.cells, c.roots.size(), f.words_per_cell,
            d_root_values.data, d_bitmap.data);
        cuda_check(cudaGetLastError(), "candidate full-update bitmap materialization");
    }

    void run_candidate_dirty(const ChangeSet &changes,
        const DeviceArray<std::uint32_t> &cells,
        const DeviceArray<std::uint32_t> &predicates,
        const DeviceArray<std::uint8_t> &values,
        const DeviceArray<std::uint32_t> &cell_offsets) {
        apply_changes(changes, cells, predicates, values);
        if (changes.cells.empty()) return;
        update_changed_root_cache_kernel<<<static_cast<unsigned>(changes.cells.size()), 128>>>(
            changes.cells.size(), cells.data, predicates.data,
            cell_offsets.data,
            d_pred_root_offsets.data, d_pred_roots.data,
            d_root_edge_offsets.data, d_root_edges.data, d_roots.data,
            c.roots.size(), f.predicates, d_predicate_work.data, d_root_values.data,
            f.words_per_cell, d_bitmap.data);
        cuda_check(cudaGetLastError(), "candidate root-local dirty update");
    }

    void reset_for_update() {
        d_predicate_work.reset_from(d_predicate_initial);
        d_bitmap.reset_from(d_initial_bitmap);
        d_root_values.reset_from(d_initial_root_values);
        cuda_check(cudaDeviceSynchronize(), "reset update buffers");
    }

    std::uint64_t native_specific_bytes() const {
        const auto max_changes = std::max(f.local_changes.cells.size(),
                                          f.latent_changes.cells.size());
        const auto max_dirty = std::max(f.local_changes.dirty_edge_count,
                                        f.latent_changes.dirty_edge_count);
        return c.nodes.size() * sizeof(condition_node_v1)
            + c.bindings.size() * sizeof(edge_condition_binding_v1)
            + c.root_node_by_edge.size() * sizeof(std::uint32_t)
            + static_cast<std::uint64_t>(f.cells) * c.nodes.size() * sizeof(std::uint8_t)
            + (static_cast<std::uint64_t>(f.cells) + 1u) * sizeof(std::uint64_t)
            + max_changes * sizeof(std::uint64_t)
            + max_dirty * sizeof(std::uint64_t)
            + sizeof(std::uint64_t)
            + sizeof(cuda_workset_status_v1);
    }

    std::uint64_t candidate_specific_bytes() const {
        return c.roots.size() * sizeof(RootPair)
            + c.edge_root.size() * sizeof(std::uint32_t)
            + c.root_edge_offsets.size() * sizeof(std::uint64_t)
            + c.root_edges.size() * sizeof(std::uint32_t)
            + c.predicate_root_offsets.size() * sizeof(std::uint64_t)
            + c.predicate_roots.size() * sizeof(std::uint32_t)
            + c.predicate_edge_offsets.size() * sizeof(std::uint64_t)
            + c.predicate_edges.size() * sizeof(std::uint32_t)
            + 2u * c.initial_root_values.size() * sizeof(std::uint8_t);
    }

    std::uint64_t shared_state_bytes() const {
        const auto change_storage = [](const ChangeSet &changes) {
            return changes.cells.size() * sizeof(std::uint32_t)
                + changes.predicates.size() * sizeof(std::uint32_t)
                + changes.values.size() * sizeof(std::uint8_t)
                + changes.cell_offsets.size() * sizeof(std::uint32_t);
        };
        return 2u * f.initial_predicates.size() * sizeof(std::uint8_t)
            + 2u * f.initial_bitmap.size() * sizeof(std::uint64_t)
            + change_storage(f.local_changes) + change_storage(f.latent_changes);
    }

    std::uint64_t joint_setup_bytes() const {
        return native_specific_bytes() + candidate_specific_bytes() + shared_state_bytes();
    }
};

TimingStats gather_samples(std::uint32_t samples, std::uint32_t warmups,
                           const std::function<Timing()> &fn) {
    TimingStats out;
    for (std::uint32_t i = 0; i < warmups; ++i) (void)fn();
    for (std::uint32_t i = 0; i < samples; ++i) {
        const auto sample = fn();
        out.device_ms.push_back(sample.device_ms);
        out.host_ms.push_back(sample.host_ms);
    }
    return out;
}

void run_fixture(const std::string &label, const std::string &directory,
                 std::uint64_t edges, std::uint32_t cells,
                 std::uint32_t genes, std::uint32_t modules,
                 std::uint32_t samples, std::uint32_t warmups,
                 bool check_only, bool ranked_presence) {
    const auto load_begin = std::chrono::steady_clock::now();
    auto fixture = load_fixture(label, directory, edges, cells, genes, modules,
                                ranked_presence);
    const auto load_end = std::chrono::steady_clock::now();
    const auto compile_begin = std::chrono::steady_clock::now();
    auto compiled = compile_condition_geometry(fixture);
    const auto initial_expected = oracle_bitmap(fixture, fixture.initial_predicates);
    fixture.initial_bitmap = initial_expected;
    const auto local_state = apply_change_oracle(fixture, fixture.local_changes);
    const auto local_expected = oracle_bitmap(fixture, local_state);
    const auto latent_state = apply_change_oracle(fixture, fixture.latent_changes);
    const auto latent_expected = oracle_bitmap(fixture, latent_state);
    const auto compile_end = std::chrono::steady_clock::now();
    Runner runner(std::move(fixture), std::move(compiled), samples, warmups);
    const auto device_setup_end = std::chrono::steady_clock::now();
    for (auto *set : {&runner.f.local_changes, &runner.f.latent_changes}) {
        std::uint64_t edge_visits = 0u, root_visits = 0u;
        for (const auto predicate : set->predicates) {
            edge_visits += runner.c.predicate_edge_offsets[predicate + 1u]
                         - runner.c.predicate_edge_offsets[predicate];
            root_visits += runner.c.predicate_root_offsets[predicate + 1u]
                         - runner.c.predicate_root_offsets[predicate];
        }
        if (edge_visits != set->dirty_edge_count)
            throw std::runtime_error("compiled predicate-edge posting fanout mismatch");
        set->root_visit_count = root_visits;
    }
    const auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::cout << std::fixed << std::setprecision(6)
              << "{\"record\":\"preparation\",\"case\":\"" << label
              << "\",\"state_view\":\""
              << (ranked_presence ? "ranked_presence" : "observed_predicates")
              << "\",\"edges\":" << edges << ",\"cells\":" << cells
              << ",\"predicates\":" << runner.f.predicates
              << ",\"roots\":" << runner.c.roots.size()
              << ",\"nodes\":" << runner.c.nodes.size()
              << ",\"load_ms\":" << ms(load_begin, load_end)
              << ",\"host_compile_ms\":" << ms(compile_begin, compile_end)
              << ",\"device_setup_ms\":" << ms(compile_end, device_setup_end)
              << ",\"shared_state_bytes\":" << runner.shared_state_bytes()
              << ",\"native_specific_bytes\":" << runner.native_specific_bytes()
              << ",\"candidate_specific_bytes\":" << runner.candidate_specific_bytes()
              << ",\"native_total_bytes\":"
              << runner.shared_state_bytes() + runner.native_specific_bytes()
              << ",\"candidate_total_bytes\":"
              << runner.shared_state_bytes() + runner.candidate_specific_bytes()
              << ",\"joint_probe_setup_bytes\":" << runner.joint_setup_bytes()
              << ",\"local_dirty_edges\":" << runner.f.local_changes.dirty_edge_count
              << ",\"latent_dirty_edges\":" << runner.f.latent_changes.dirty_edge_count
              << ",\"local_update_items\":" << runner.f.local_changes.cells.size()
              << ",\"local_effective_state_changes\":"
              << runner.f.local_changes.effective_state_change_count
              << ",\"latent_update_items\":" << runner.f.latent_changes.cells.size()
              << ",\"latent_effective_state_changes\":"
              << runner.f.latent_changes.effective_state_change_count
              << ",\"local_root_visits\":" << runner.f.local_changes.root_visit_count
              << ",\"latent_root_visits\":" << runner.f.latent_changes.root_visit_count
              << "}\n";

    // Correctness for each forward path is checked against the independent scalar oracle.
    for (const auto path : {0, 1, 2}) {
        const auto timing = runner.measure([&] {
            if (path == 0) runner.run_native_full(false);
            else if (path == 1) runner.run_native_full(true);
            else runner.run_candidate_full();
        });
        runner.check_native_status(path == 0 ? "native-full" : "prepared-full");
        require_bitmap_equal(runner.copy_bitmap(), initial_expected,
            path == 0 ? "native full binding search"
            : path == 1 ? "prepared direct map full" : "root-cache full");
        (void)timing;
    }
    if (check_only) {
        std::cout << "{\"record\":\"correctness\",\"case\":\"" << label
                  << "\",\"status\":\"pass\",\"full_paths\":3}\n";
    } else {
        std::array<TimingStats, 3> stats;
        for (std::uint32_t i = 0; i < warmups; ++i)
            for (int path = 0; path < 3; ++path) {
                runner.measure([&] {
                    if (path == 0) runner.run_native_full(false);
                    else if (path == 1) runner.run_native_full(true);
                    else runner.run_candidate_full();
                });
            }
        for (std::uint32_t sample = 0; sample < samples; ++sample) {
            for (int step = 0; step < 3; ++step) {
                const int path = static_cast<int>((sample + step) % 3u);
                const auto timing = runner.measure([&] {
                    if (path == 0) runner.run_native_full(false);
                    else if (path == 1) runner.run_native_full(true);
                    else runner.run_candidate_full();
                });
                stats[path].device_ms.push_back(timing.device_ms);
                stats[path].host_ms.push_back(timing.host_ms);
                runner.check_native_status(path == 0 ? "native-full" : "prepared-full");
            }
        }
        summarize(stats[0], label, "native_binding_search", "full", label);
        summarize(stats[1], label, "prepared_edge_root_map", "full", label);
        summarize(stats[2], label, "root_pair_cache", "full", label);
    }

    auto benchmark_update = [&](const ChangeSet &changes, bool local,
                                const std::vector<std::uint64_t> &expected,
                                const std::string &scenario) {
        auto &change_cells = local ? runner.d_local_cells : runner.d_latent_cells;
        auto &change_predicates = local ? runner.d_local_predicates : runner.d_latent_predicates;
        auto &change_values = local ? runner.d_local_values : runner.d_latent_values;
        auto &change_offsets = local ? runner.d_local_cell_offsets : runner.d_latent_cell_offsets;
        enum Arm { full_search, full_prepared, dirty_search, dirty_prepared,
                   cache_full, cache_dirty, arm_count };
        const char *names[arm_count] = {"native_full_binding_search", "native_full_direct_map",
            "native_dirty_binding_search", "native_dirty_direct_map",
            "root_cache_full_rebuild", "root_local_predicate_postings"};
        auto run_arm = [&](int arm) {
            runner.reset_for_update();
            return runner.measure([&] {
                switch (arm) {
                case full_search:
                    runner.run_native_full_update(changes, change_cells,
                        change_predicates, change_values, false); break;
                case full_prepared:
                    runner.run_native_full_update(changes, change_cells,
                        change_predicates, change_values, true); break;
                case dirty_search:
                    runner.run_native_dirty(changes, change_cells,
                        change_predicates, change_values, change_offsets, false); break;
                case dirty_prepared:
                    runner.run_native_dirty(changes, change_cells,
                        change_predicates, change_values, change_offsets, true); break;
                case cache_full:
                    runner.run_candidate_full_update(changes, change_cells,
                        change_predicates, change_values); break;
                case cache_dirty:
                    runner.run_candidate_dirty(changes, change_cells,
                        change_predicates, change_values, change_offsets); break;
                default: throw std::runtime_error("bad benchmark arm");
                }
            });
        };
        for (int arm = 0; arm < arm_count; ++arm) {
            const auto timing = run_arm(arm);
            if (arm < 4) runner.check_native_status(names[arm]);
            require_bitmap_equal(runner.copy_bitmap(), expected,
                scenario + "/" + names[arm]);
            (void)timing;
        }
        if (check_only) {
            std::cout << "{\"record\":\"correctness\",\"case\":\"" << label
                      << "\",\"update\":\"" << scenario
                      << "\",\"status\":\"pass\",\"paths\":" << arm_count << "}\n";
            return;
        }
        std::array<TimingStats, arm_count> stats;
        for (std::uint32_t i = 0; i < warmups; ++i)
            for (int arm = 0; arm < arm_count; ++arm) (void)run_arm(arm);
        for (std::uint32_t sample = 0; sample < samples; ++sample) {
            for (int step = 0; step < arm_count; ++step) {
                const int arm = static_cast<int>((sample + step) % arm_count);
                const auto timing = run_arm(arm);
                stats[arm].device_ms.push_back(timing.device_ms);
                stats[arm].host_ms.push_back(timing.host_ms);
                if (arm < 4) runner.check_native_status(names[arm]);
            }
        }
        for (int arm = 0; arm < arm_count; ++arm)
            summarize(stats[arm], label, names[arm], scenario, label);
        // Per-stage probes make dirty construction and root-local work visible.
        const auto worklist = gather_samples(std::min(samples, 7u), 2u, [&] {
            return runner.measure([&] { runner.build_dirty(changes,
                change_predicates, change_offsets); });
        });
        summarize(worklist, label, "native_dirty_worklist_build", scenario + "_component", label);
        const auto root_update = gather_samples(std::min(samples, 7u), 2u, [&] {
            runner.reset_for_update();
            runner.apply_changes(changes, change_cells, change_predicates, change_values);
            return runner.measure([&] {
                if (!changes.cells.empty()) {
                    update_changed_root_cache_kernel<<<static_cast<unsigned>(changes.cells.size()), 128>>>(
                        changes.cells.size(), change_cells.data, change_predicates.data,
                        change_offsets.data,
                        runner.d_pred_root_offsets.data, runner.d_pred_roots.data,
                        runner.d_root_edge_offsets.data, runner.d_root_edges.data,
                        runner.d_roots.data, runner.c.roots.size(), runner.f.predicates,
                        runner.d_predicate_work.data, runner.d_root_values.data,
                        runner.f.words_per_cell, runner.d_bitmap.data);
                    cuda_check(cudaGetLastError(), "candidate root-local component");
                }
            });
        });
        summarize(root_update, label, "root_local_posting_update", scenario + "_component", label);
    };

    benchmark_update(runner.f.local_changes, true, local_expected, "local_gene_patch");
    benchmark_update(runner.f.latent_changes, false, latent_expected, "shared_latent_toggle");

    std::cout << "{\"record\":\"correctness\",\"case\":\"" << label
              << "\",\"status\":\"pass\",\"all\":true}\n";
}

struct Options {
    std::string label;
    std::string directory;
    std::uint64_t edges = 0u;
    std::uint32_t cells = 0u, genes = 0u, modules = 0u;
    std::uint32_t samples = 11u, warmups = 2u;
    bool check_only = false, ranked_presence = false;
};

Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--check-only") o.check_only = true;
        else if (arg == "--ranked-presence") o.ranked_presence = true;
        else if (arg == "--label" && i + 1 < argc) o.label = argv[++i];
        else if (arg == "--fixture-dir" && i + 1 < argc) o.directory = argv[++i];
        else if (arg == "--edges" && i + 1 < argc) o.edges = std::stoull(argv[++i]);
        else if (arg == "--cells" && i + 1 < argc) o.cells = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--genes" && i + 1 < argc) o.genes = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--modules" && i + 1 < argc) o.modules = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--samples" && i + 1 < argc) o.samples = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--warmups" && i + 1 < argc) o.warmups = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else throw std::runtime_error("unknown or incomplete option: " + arg);
    }
    if (o.label.empty() || o.directory.empty() || !o.edges || !o.cells || !o.genes || !o.modules)
        throw std::runtime_error("required: --label --fixture-dir --edges --cells --genes --modules");
    return o;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto options = parse(argc, argv);
        run_fixture(options.label, options.directory, options.edges, options.cells,
            options.genes, options.modules, options.samples, options.warmups,
            options.check_only, options.ranked_presence);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "scaled_condition_probe: " << error.what() << '\n';
        return 1;
    }
}
