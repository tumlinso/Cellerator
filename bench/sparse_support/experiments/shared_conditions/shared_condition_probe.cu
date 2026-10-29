#include <Cellerator/compute/gating/cuda_worksets.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace gate = cellerator::compute::gating;
namespace indexing = cellerator::compute::indexing;

#define CUDA_CHECK(expr) do { \
    const auto cuda_status = (expr); \
    if (cuda_status != cudaSuccess) \
        throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(cuda_status)); \
} while (0)

namespace {

template <class T> class device_buffer {
public:
    device_buffer() = default;
    explicit device_buffer(std::size_t count) : count_(count) {
        if (count_) CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), count_ * sizeof(T)));
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    device_buffer(device_buffer &&other) noexcept
        : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }
    device_buffer &operator=(device_buffer &&other) noexcept {
        if (this != &other) {
            if (data_) cudaFree(data_);
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    ~device_buffer() { if (data_) cudaFree(data_); }
    T *get() const { return data_; }
    std::size_t size() const { return count_; }
private:
    T *data_ = nullptr;
    std::size_t count_ = 0;
};

template <class T> void upload(device_buffer<T> &dst, const std::vector<T> &src) {
    if (src.size() > dst.size()) throw std::runtime_error("device upload overflow");
    if (!src.empty()) CUDA_CHECK(cudaMemcpy(dst.get(), src.data(), src.size() * sizeof(T),
                                            cudaMemcpyHostToDevice));
}

template <class T> std::vector<T> download(const device_buffer<T> &src,
                                           std::size_t count) {
    if (count > src.size()) throw std::runtime_error("device download overflow");
    std::vector<T> out(count);
    if (count) CUDA_CHECK(cudaMemcpy(out.data(), src.get(), count * sizeof(T),
                                     cudaMemcpyDeviceToHost));
    return out;
}

__global__ void transpose_leaf_planes(
    const gate::condition_node_v1 *nodes, const std::uint32_t *leaf_ids,
    std::uint64_t leaf_count, std::uint64_t cells, std::uint32_t predicates_count,
    const std::uint8_t *predicates, std::uint32_t groups, std::uint32_t *planes) {
    const auto group = static_cast<std::uint32_t>(blockIdx.x);
    const auto lane = threadIdx.x & 31u;
    const auto leaf_at = static_cast<std::uint64_t>(blockIdx.y) * 4u
                       + threadIdx.x / 32u;
    if (group >= groups) return;
    const bool valid_leaf = leaf_at < leaf_count;
    const auto node_id = valid_leaf ? leaf_ids[leaf_at] : 0u;
    const auto &node = nodes[node_id];
    const auto cell = static_cast<std::uint64_t>(group) * 32u + lane;
    const bool value = cell < cells
        && valid_leaf
        && predicates[cell * predicates_count + node.predicate_index] == node.expected_value;
    const auto bits = __ballot_sync(0xffffffffu, value);
    if (lane == 0 && valid_leaf)
        planes[static_cast<std::uint64_t>(node_id) * groups + group] = bits;
}

__device__ __forceinline__ std::uint32_t valid_cell_mask(
    std::uint32_t group, std::uint32_t groups, std::uint64_t cells) {
    if (group + 1u < groups || cells % 32u == 0u) return 0xffffffffu;
    return (std::uint32_t{1} << (cells % 32u)) - 1u;
}

__device__ __forceinline__ bool evaluate_scalar_nodes(
    const gate::condition_program_view_v1 &p, const std::uint8_t *predicates,
    std::uint8_t *scratch) {
    for (std::uint64_t i = 0; i < p.node_count; ++i) {
        const auto &n = p.nodes[i];
        switch (n.opcode) {
        case gate::condition_opcode_v1::predicate:
            scratch[i] = predicates[n.predicate_index] == n.expected_value;
            break;
        case gate::condition_opcode_v1::all_of: {
            bool value = true;
            for (std::uint32_t j = 0; j < n.child_count; ++j)
                value = value && scratch[n.first_child + j] != 0;
            scratch[i] = value;
            break;
        }
        case gate::condition_opcode_v1::any_of: {
            bool value = false;
            for (std::uint32_t j = 0; j < n.child_count; ++j)
                value = value || scratch[n.first_child + j] != 0;
            scratch[i] = value;
            break;
        }
        case gate::condition_opcode_v1::logical_not:
            scratch[i] = scratch[n.first_child] == 0;
            break;
        default:
            return false;
        }
    }
    return true;
}

__global__ void evaluate_condition_plane_level(
    const gate::condition_node_v1 *nodes, const std::uint32_t *node_ids,
    std::uint64_t node_count, std::uint64_t cells, std::uint32_t groups,
    const std::uint32_t *planes, std::uint32_t *output_planes) {
    const auto group = static_cast<std::uint32_t>(blockIdx.x);
    const auto node_at = static_cast<std::uint64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (group >= groups || node_at >= node_count) return;
    const auto node_id = node_ids[node_at];
    const auto &node = nodes[node_id];
    const auto valid = valid_cell_mask(group, groups, cells);
    std::uint32_t value = node.opcode == gate::condition_opcode_v1::all_of
        ? valid : 0u;
    for (std::uint32_t i = 0; i < node.child_count; ++i) {
        const auto child = planes[static_cast<std::uint64_t>(node.first_child + i) * groups + group];
        if (node.opcode == gate::condition_opcode_v1::all_of) value &= child;
        else if (node.opcode == gate::condition_opcode_v1::any_of) value |= child;
        else if (node.opcode == gate::condition_opcode_v1::logical_not) value = ~child;
    }
    output_planes[static_cast<std::uint64_t>(node_id) * groups + group] = value & valid;
}

__global__ void materialize_full_bitmaps(
    const std::uint32_t *root_by_edge, std::uint64_t edges,
    std::uint64_t cells, std::uint32_t groups, const std::uint32_t *planes,
    std::uint64_t *bitmaps) {
    const auto cell = static_cast<std::uint64_t>(blockIdx.x);
    if (cell >= cells) return;
    const auto cell_word = static_cast<std::uint32_t>(cell / 32u);
    const auto cell_bit = static_cast<std::uint32_t>(cell % 32u);
    auto *row = bitmaps + cell * (edges / 64u + (edges % 64u != 0u));
    for (std::uint64_t edge = threadIdx.x; edge < edges; edge += blockDim.x) {
        const auto root = root_by_edge[edge];
        const bool pass = root == UINT32_MAX
            || ((planes[static_cast<std::uint64_t>(root) * groups + cell_word]
                  >> cell_bit) & 1u) != 0;
        if (pass) atomicOr(reinterpret_cast<unsigned long long *>(&row[edge / 64u]),
                           static_cast<unsigned long long>(std::uint64_t{1} << (edge % 64u)));
    }
}

__global__ void direct_map_full_bitmaps(
    gate::condition_program_view_v1 program, std::uint64_t cells,
    const std::uint8_t *predicates, const std::uint32_t *root_by_edge,
    std::uint64_t *bitmaps, std::uint8_t *node_scratch) {
    const auto cell = static_cast<std::uint64_t>(blockIdx.x);
    if (cell >= cells) return;
    __shared__ int valid;
    auto *nodes = node_scratch + cell * program.node_count;
    const auto *row_predicates = predicates + cell * program.predicate_count;
    if (threadIdx.x == 0) valid = evaluate_scalar_nodes(program, row_predicates, nodes);
    __syncthreads();
    if (!valid) return;
    const auto words = program.edge_count / 64u + (program.edge_count % 64u != 0u);
    auto *row = bitmaps + cell * words;
    for (std::uint64_t word = threadIdx.x; word < words; word += blockDim.x) row[word] = 0;
    __syncthreads();
    for (std::uint64_t edge = threadIdx.x; edge < program.edge_count; edge += blockDim.x) {
        const auto root = root_by_edge[edge];
        if (root == UINT32_MAX || nodes[root] != 0)
            atomicOr(reinterpret_cast<unsigned long long *>(&row[edge / 64u]),
                static_cast<unsigned long long>(std::uint64_t{1} << (edge % 64u)));
    }
}

__global__ void update_dirty_bitmaps(
    const std::uint32_t *root_by_edge, const std::uint64_t *dirty_offsets,
    const std::uint64_t *dirty_edges, std::uint64_t cells, std::uint64_t edges,
    std::uint32_t groups, const std::uint32_t *planes, std::uint64_t *bitmaps) {
    const auto cell = static_cast<std::uint64_t>(blockIdx.x);
    if (cell >= cells) return;
    const auto begin = dirty_offsets[cell], end = dirty_offsets[cell + 1u];
    const auto cell_word = static_cast<std::uint32_t>(cell / 32u);
    const auto cell_bit = static_cast<std::uint32_t>(cell % 32u);
    const auto words = edges / 64u + (edges % 64u != 0u);
    auto *row = bitmaps + cell * words;
    for (auto i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const auto edge = dirty_edges[i];
        const auto root = root_by_edge[edge];
        const bool pass = root == UINT32_MAX
            || ((planes[static_cast<std::uint64_t>(root) * groups + cell_word]
                  >> cell_bit) & 1u) != 0;
        const auto bit = std::uint64_t{1} << (edge % 64u);
        auto *word = reinterpret_cast<unsigned long long *>(&row[edge / 64u]);
        if (pass) atomicOr(word, static_cast<unsigned long long>(bit));
        else atomicAnd(word, static_cast<unsigned long long>(~bit));
    }
}

__global__ void direct_map_dirty_bitmaps(
    gate::condition_program_view_v1 program, std::uint64_t cells,
    const std::uint8_t *predicates, const std::uint64_t *dirty_offsets,
    const std::uint64_t *dirty_edges, const std::uint32_t *root_by_edge,
    std::uint64_t *bitmaps, std::uint8_t *node_scratch) {
    const auto cell = static_cast<std::uint64_t>(blockIdx.x);
    if (cell >= cells) return;
    const auto begin = dirty_offsets[cell], end = dirty_offsets[cell + 1u];
    __shared__ int valid;
    auto *nodes = node_scratch + cell * program.node_count;
    if (threadIdx.x == 0) {
        valid = end >= begin;
        if (valid && begin != end)
            valid = evaluate_scalar_nodes(program,
                predicates + cell * program.predicate_count, nodes);
    }
    __syncthreads();
    if (!valid || begin == end) return;
    const auto words = program.edge_count / 64u + (program.edge_count % 64u != 0u);
    auto *row = bitmaps + cell * words;
    for (auto i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const auto edge = dirty_edges[i];
        const auto root = root_by_edge[edge];
        const bool pass = root == UINT32_MAX || nodes[root] != 0;
        const auto bit = std::uint64_t{1} << (edge % 64u);
        auto *word = reinterpret_cast<unsigned long long *>(&row[edge / 64u]);
        if (pass) atomicOr(word, static_cast<unsigned long long>(bit));
        else atomicAnd(word, static_cast<unsigned long long>(~bit));
    }
}

struct options {
    std::uint64_t edges = 2176;
    std::uint64_t cells = 64;
    std::uint64_t seed = 0x5a17c9e3u;
    std::uint32_t repeats = 11;
    std::string sharing = "few";
    bool check_only = false;
};

struct fixture {
    options cfg;
    std::uint32_t predicate_count = 0;
    std::vector<gate::condition_node_v1> nodes;
    std::vector<gate::edge_condition_binding_v1> bindings;
    gate::condition_program_view_v1 program{};
    std::vector<std::uint32_t> root_by_edge;
    std::vector<std::vector<std::uint32_t>> composite_levels;
    std::vector<std::uint32_t> leaf_nodes;
    std::vector<std::vector<std::uint32_t>> changed_composite_levels;
    std::vector<std::uint32_t> changed_leaf_nodes;
    std::vector<std::uint8_t> initial_predicates;
    std::vector<std::uint8_t> changed_predicates;
    std::vector<std::uint64_t> dirty_offsets;
    std::vector<std::uint64_t> dirty_edges;
    std::vector<indexing::indexed_work_item_v1> primal_items;
    std::vector<indexing::indexed_work_item_v1> derivative_items;
    std::vector<std::uint64_t> primal_offsets;
    std::vector<std::uint64_t> derivative_offsets;
    std::vector<bool> depends_on_changed;
    std::uint64_t digest = 1469598103934665603ull;
};

std::uint64_t next_random(std::uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 2685821657736338717ull;
}

fixture make_fixture(const options &cfg) {
    fixture f;
    f.cfg = cfg;
    f.predicate_count = cfg.sharing == "few" ? 4u : 128u;
    if (cfg.sharing != "few" && cfg.sharing != "weak")
        throw std::runtime_error("--sharing must be few or weak");

    using gate::condition_opcode_v1;
    if (cfg.sharing == "few") {
        f.nodes = {
            {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 0u, 0u},
            {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 1u, 0u},
            {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 2u, 0u},
            {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 3u, 0u},
            {condition_opcode_v1::all_of, 1u, 0u, 0u, 2u, 0u, 0u},
            {condition_opcode_v1::any_of, 1u, 0u, 2u, 2u, 0u, 0u},
            {condition_opcode_v1::all_of, 1u, 0u, 0u, 3u, 0u, 0u},
        };
        f.root_by_edge.resize(cfg.edges);
        const auto dirty_fraction = 1.0 / 16.0;
        const auto dirty_prefix = static_cast<std::uint64_t>(
            std::max(1.0, std::ceil(static_cast<double>(cfg.edges) * dirty_fraction)));
        for (std::uint64_t edge = 0; edge < cfg.edges; ++edge) {
            std::uint32_t root;
            if (edge < dirty_prefix)
                root = edge % 3u == 0u ? 0u : edge % 3u == 1u ? 4u : 6u;
            else {
                const std::array<std::uint32_t, 4> roots{{1u, 2u, 3u, 5u}};
                root = roots[(edge - dirty_prefix) % roots.size()];
            }
            f.root_by_edge[edge] = root;
        }
    } else {
        f.nodes.reserve(f.predicate_count + cfg.edges);
        for (std::uint32_t p = 0; p < f.predicate_count; ++p)
            f.nodes.push_back({condition_opcode_v1::predicate, 1u, 0u, 0u,
                               0u, p, 0u});
        f.root_by_edge.resize(cfg.edges);
        // Every edge gets its own conjunction over a distinct contiguous
        // predicate window. The supports share predicate leaves but almost no
        // edge-level condition roots, making this a deliberately weak-sharing
        // case for the plane cache.
        std::uint32_t start = 0u, width = 2u;
        for (std::uint64_t edge = 0; edge < cfg.edges; ++edge) {
            if (width > f.predicate_count) throw std::runtime_error("weak fixture exhausted unique roots");
            const auto root = static_cast<std::uint32_t>(f.nodes.size());
            f.nodes.push_back({condition_opcode_v1::all_of, 1u, 0u,
                               start, width, 0u, 0u});
            f.root_by_edge[edge] = root;
            ++start;
            if (start + width > f.predicate_count) {
                ++width;
                start = 0u;
            }
        }
    }

    f.bindings.reserve(cfg.edges);
    for (std::uint64_t edge = 0; edge < cfg.edges; ++edge)
        f.bindings.push_back({edge, f.root_by_edge[edge], 0u});
    f.program = {1u, f.nodes.data(), f.nodes.size(), f.bindings.data(),
                 f.bindings.size(), cfg.edges, f.predicate_count, 0u};
    if (gate::validate_condition_program_v1(f.program) != gate::condition_status_v1::success)
        throw std::runtime_error("condition fixture validation failed");

    std::vector<std::uint32_t> depth(f.nodes.size(), 0u);
    f.depends_on_changed.resize(f.nodes.size(), false);
    for (std::uint32_t i = 0; i < f.nodes.size(); ++i) {
        const auto &n = f.nodes[i];
        if (n.opcode == condition_opcode_v1::predicate) {
            f.leaf_nodes.push_back(i);
            f.depends_on_changed[i] = n.predicate_index == 0u;
        } else {
            for (std::uint32_t j = 0; j < n.child_count; ++j) {
                const auto child = n.first_child + j;
                depth[i] = std::max(depth[i], depth[child] + 1u);
                f.depends_on_changed[i] = f.depends_on_changed[i]
                    || f.depends_on_changed[child];
            }
        }
    }
    const auto max_depth = *std::max_element(depth.begin(), depth.end());
    f.composite_levels.resize(max_depth + 1u);
    f.changed_composite_levels.resize(max_depth + 1u);
    for (std::uint32_t i = 0; i < f.nodes.size(); ++i) {
        if (f.nodes[i].opcode == condition_opcode_v1::predicate) {
            if (f.nodes[i].predicate_index == 0u) f.changed_leaf_nodes.push_back(i);
        } else {
            f.composite_levels[depth[i]].push_back(i);
            if (f.depends_on_changed[i]) f.changed_composite_levels[depth[i]].push_back(i);
        }
    }

    f.initial_predicates.resize(cfg.cells * f.predicate_count);
    f.changed_predicates.resize(f.initial_predicates.size());
    auto rng = cfg.seed;
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        for (std::uint32_t p = 0; p < f.predicate_count; ++p)
            f.initial_predicates[cell * f.predicate_count + p]
                = static_cast<std::uint8_t>((next_random(rng) >> 63u) & 1u);
        f.initial_predicates[cell * f.predicate_count] ^= 1u;
    }
    f.changed_predicates = f.initial_predicates;
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell)
        f.changed_predicates[cell * f.predicate_count] ^= 1u;

    std::vector<bool> dirty(f.nodes.size(), false);
    for (std::uint32_t i = 0; i < f.nodes.size(); ++i)
        dirty[i] = f.depends_on_changed[i];
    f.dirty_offsets.resize(cfg.cells + 1u);
    std::vector<std::uint64_t> dirty_for_cell;
    for (std::uint64_t edge = 0; edge < cfg.edges; ++edge)
        if (dirty[f.root_by_edge[edge]]) dirty_for_cell.push_back(edge);
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        f.dirty_offsets[cell] = f.dirty_edges.size();
        f.dirty_edges.insert(f.dirty_edges.end(), dirty_for_cell.begin(), dirty_for_cell.end());
    }
    f.dirty_offsets[cfg.cells] = f.dirty_edges.size();

    // Deterministic sparse proposal lists: primal and derivative candidates
    // are disjoint 1/32 edge samples, independently consumed by production's
    // condition workset compactor in all timing paths.
    f.primal_offsets.resize(cfg.cells + 1u);
    f.derivative_offsets.resize(cfg.cells + 1u);
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        f.primal_offsets[cell] = f.primal_items.size();
        f.derivative_offsets[cell] = f.derivative_items.size();
        for (std::uint64_t edge = 0; edge < cfg.edges; ++edge) {
            if ((edge % 32u) == 0u)
                f.primal_items.push_back({edge + 1u, edge});
            else if ((edge % 32u) == 1u)
                f.derivative_items.push_back({edge + 1u, edge});
        }
    }
    f.primal_offsets[cfg.cells] = f.primal_items.size();
    f.derivative_offsets[cfg.cells] = f.derivative_items.size();
    return f;
}

std::vector<std::uint8_t> scalar_nodes(const fixture &f,
                                       const std::uint8_t *predicates) {
    std::vector<std::uint8_t> values(f.nodes.size(), 0u);
    for (std::size_t i = 0; i < f.nodes.size(); ++i) {
        const auto &n = f.nodes[i];
        switch (n.opcode) {
        case gate::condition_opcode_v1::predicate:
            values[i] = predicates[n.predicate_index] == n.expected_value;
            break;
        case gate::condition_opcode_v1::all_of:
            values[i] = 1u;
            for (std::uint32_t j = 0; j < n.child_count; ++j)
                values[i] = values[i] && values[n.first_child + j];
            break;
        case gate::condition_opcode_v1::any_of:
            for (std::uint32_t j = 0; j < n.child_count; ++j)
                values[i] = values[i] || values[n.first_child + j];
            break;
        case gate::condition_opcode_v1::logical_not:
            values[i] = !values[n.first_child];
            break;
        }
    }
    return values;
}

std::vector<std::uint64_t> scalar_bitmap(const fixture &f,
                                         const std::vector<std::uint8_t> &predicates) {
    const auto words = (f.cfg.edges + 63u) / 64u;
    std::vector<std::uint64_t> bits(f.cfg.cells * words, 0u);
    for (std::uint64_t cell = 0; cell < f.cfg.cells; ++cell) {
        const auto values = scalar_nodes(f, predicates.data() + cell * f.predicate_count);
        for (std::uint64_t edge = 0; edge < f.cfg.edges; ++edge)
            if (values[f.root_by_edge[edge]])
                bits[cell * words + edge / 64u] |= std::uint64_t{1} << (edge % 64u);
    }
    return bits;
}

struct device_inputs {
    device_buffer<gate::condition_node_v1> nodes;
    device_buffer<gate::edge_condition_binding_v1> bindings;
    device_buffer<std::uint32_t> root_by_edge;
    device_buffer<std::uint8_t> initial_predicates;
    device_buffer<std::uint8_t> changed_predicates;
    device_buffer<std::uint64_t> dirty_offsets;
    device_buffer<std::uint64_t> dirty_edges;
    device_buffer<std::uint32_t> leaf_nodes;
    device_buffer<std::uint32_t> changed_leaf_nodes;
    std::vector<device_buffer<std::uint32_t>> composite_levels;
    std::vector<device_buffer<std::uint32_t>> changed_composite_levels;
    std::uint32_t groups = 0;
    gate::condition_program_view_v1 program{};

    explicit device_inputs(const fixture &f,
                           const std::vector<std::uint32_t> &compiled_roots)
        : nodes(f.nodes.size()), bindings(f.bindings.size()), root_by_edge(compiled_roots.size()),
          initial_predicates(f.initial_predicates.size()),
          changed_predicates(f.changed_predicates.size()), dirty_offsets(f.dirty_offsets.size()),
          dirty_edges(f.dirty_edges.size()), leaf_nodes(f.leaf_nodes.size()),
          changed_leaf_nodes(f.changed_leaf_nodes.size()),
          groups(static_cast<std::uint32_t>((f.cfg.cells + 31u) / 32u)) {
        upload(nodes, f.nodes);
        upload(bindings, f.bindings);
        upload(root_by_edge, compiled_roots);
        upload(initial_predicates, f.initial_predicates);
        upload(changed_predicates, f.changed_predicates);
        upload(dirty_offsets, f.dirty_offsets);
        upload(dirty_edges, f.dirty_edges);
        upload(leaf_nodes, f.leaf_nodes);
        upload(changed_leaf_nodes, f.changed_leaf_nodes);
        composite_levels.reserve(f.composite_levels.size());
        changed_composite_levels.reserve(f.changed_composite_levels.size());
        for (const auto &level : f.composite_levels) {
            composite_levels.emplace_back(level.size());
            upload(composite_levels.back(), level);
        }
        for (const auto &level : f.changed_composite_levels) {
            changed_composite_levels.emplace_back(level.size());
            upload(changed_composite_levels.back(), level);
        }
        program = f.program;
        program.nodes = nodes.get();
        program.edges = bindings.get();
    }
};

struct workset_buffers {
    std::uint64_t primal_per_cell = 0;
    std::uint64_t derivative_per_cell = 0;
    device_buffer<std::uint64_t> primal_input_offsets;
    device_buffer<std::uint64_t> derivative_input_offsets;
    device_buffer<indexing::indexed_work_item_v1> primal_input_items;
    device_buffer<indexing::indexed_work_item_v1> derivative_input_items;
    device_buffer<std::uint64_t> primal_output_offsets;
    device_buffer<std::uint64_t> derivative_output_offsets;
    device_buffer<indexing::indexed_work_item_v1> primal_output_items;
    device_buffer<indexing::indexed_work_item_v1> derivative_output_items;
    device_buffer<std::uint64_t> primal_stage_offsets;
    device_buffer<std::uint64_t> derivative_stage_offsets;
    device_buffer<indexing::indexed_work_item_v1> primal_stage_items;
    device_buffer<indexing::indexed_work_item_v1> derivative_stage_items;
    device_buffer<std::uint64_t> primal_counts;
    device_buffer<std::uint64_t> derivative_counts;
    device_buffer<gate::cuda_workset_status_v1> status;

    workset_buffers(const fixture &f)
        : primal_per_cell(f.primal_items.size() / f.cfg.cells),
          derivative_per_cell(f.derivative_items.size() / f.cfg.cells),
          primal_input_offsets(f.primal_offsets.size()), derivative_input_offsets(f.derivative_offsets.size()),
          primal_input_items(f.primal_items.size()), derivative_input_items(f.derivative_items.size()),
          primal_output_offsets(f.cfg.cells + 1u), derivative_output_offsets(f.cfg.cells + 1u),
          primal_output_items(f.primal_items.size()),
          derivative_output_items(f.primal_items.size() + f.derivative_items.size()),
          primal_stage_offsets(f.cfg.cells + 1u), derivative_stage_offsets(f.cfg.cells + 1u),
          primal_stage_items(f.primal_items.size()),
          derivative_stage_items(f.primal_items.size() + f.derivative_items.size()),
          primal_counts(f.cfg.cells), derivative_counts(f.cfg.cells), status(1u) {
        upload(primal_input_offsets, f.primal_offsets);
        upload(derivative_input_offsets, f.derivative_offsets);
        upload(primal_input_items, f.primal_items);
        upload(derivative_input_items, f.derivative_items);
        std::vector<std::uint64_t> p_stage(f.cfg.cells + 1u), d_stage(f.cfg.cells + 1u);
        for (std::uint64_t cell = 0; cell <= f.cfg.cells; ++cell) {
            p_stage[cell] = cell * primal_per_cell;
            d_stage[cell] = cell * (primal_per_cell + derivative_per_cell);
        }
        upload(primal_stage_offsets, p_stage);
        upload(derivative_stage_offsets, d_stage);
    }

    void build(const gate::condition_program_view_v1 &program, const fixture &f,
               const std::uint64_t *bitmaps) {
        gate::cuda_candidate_csr_v1 primal{
            primal_input_offsets.get(), primal_input_items.get(),
            static_cast<std::uint64_t>(primal_input_items.size())};
        gate::cuda_candidate_csr_v1 derivative{
            derivative_input_offsets.get(), derivative_input_items.get(),
            static_cast<std::uint64_t>(derivative_input_items.size())};
        gate::cuda_output_csr_v1 p_out{
            primal_output_offsets.get(), static_cast<std::uint64_t>(primal_output_offsets.size()),
            primal_output_items.get(), static_cast<std::uint64_t>(primal_output_items.size())};
        gate::cuda_output_csr_v1 d_out{
            derivative_output_offsets.get(), static_cast<std::uint64_t>(derivative_output_offsets.size()),
            derivative_output_items.get(), static_cast<std::uint64_t>(derivative_output_items.size())};
        gate::cuda_workset_scratch_v1 scratch{};
        scratch.primal_offsets = primal_stage_offsets.get();
        scratch.primal_items = primal_stage_items.get();
        scratch.primal_item_capacity = primal_stage_items.size();
        scratch.derivative_offsets = derivative_stage_offsets.get();
        scratch.derivative_items = derivative_stage_items.get();
        scratch.derivative_item_capacity = derivative_stage_items.size();
        scratch.primal_counts = primal_counts.get();
        scratch.derivative_counts = derivative_counts.get();
        scratch.status = status.get();
        scratch.primal_offsets_capacity = primal_stage_offsets.size();
        scratch.derivative_offsets_capacity = derivative_stage_offsets.size();
        CUDA_CHECK(gate::cuda_build_condition_worksets_v1(program, f.cfg.cells, bitmaps,
            f.cfg.cells * ((f.cfg.edges + 63u) / 64u), primal, derivative, p_out, d_out,
            scratch, nullptr));
    }
};

void build_candidate_full_planes(const fixture &f, const device_inputs &d,
                                 const std::uint8_t *predicates,
                                 std::uint32_t *planes) {
    const dim3 leaf_grid(d.groups, static_cast<unsigned>((f.leaf_nodes.size() + 3u) / 4u));
    transpose_leaf_planes<<<leaf_grid, 128>>>(d.nodes.get(), d.leaf_nodes.get(),
        f.leaf_nodes.size(), f.cfg.cells, f.predicate_count, predicates, d.groups, planes);
    CUDA_CHECK(cudaGetLastError());
    for (std::size_t level = 0; level < f.composite_levels.size(); ++level) {
        if (f.composite_levels[level].empty()) continue;
        const dim3 grid(d.groups, static_cast<unsigned>((f.composite_levels[level].size() + 127u) / 128u));
        evaluate_condition_plane_level<<<grid, 128>>>(d.nodes.get(), d.composite_levels[level].get(),
            f.composite_levels[level].size(), f.cfg.cells, d.groups, planes, planes);
        CUDA_CHECK(cudaGetLastError());
    }
}

void candidate_changed_planes(const fixture &f, const device_inputs &d,
                              const std::uint8_t *predicates, std::uint32_t *planes) {
    const dim3 leaf_grid(d.groups, static_cast<unsigned>((f.changed_leaf_nodes.size() + 3u) / 4u));
    if (!f.changed_leaf_nodes.empty()) {
        transpose_leaf_planes<<<leaf_grid, 128>>>(d.nodes.get(), d.changed_leaf_nodes.get(),
            f.changed_leaf_nodes.size(),
            f.cfg.cells, f.predicate_count, predicates, d.groups, planes);
        CUDA_CHECK(cudaGetLastError());
    }
    for (std::size_t level = 0; level < f.changed_composite_levels.size(); ++level) {
        if (f.changed_composite_levels[level].empty()) continue;
        const dim3 grid(d.groups,
            static_cast<unsigned>((f.changed_composite_levels[level].size() + 127u) / 128u));
        evaluate_condition_plane_level<<<grid, 128>>>(d.nodes.get(),
            d.changed_composite_levels[level].get(), f.changed_composite_levels[level].size(),
            f.cfg.cells, d.groups, planes, planes);
        CUDA_CHECK(cudaGetLastError());
    }
}

void candidate_materialize(const fixture &f, const device_inputs &d,
                           std::uint32_t *planes, std::uint64_t *bitmaps,
                           bool clear_first) {
    const auto words = f.cfg.cells * ((f.cfg.edges + 63u) / 64u);
    if (clear_first) CUDA_CHECK(cudaMemsetAsync(bitmaps, 0, words * sizeof(std::uint64_t)));
    materialize_full_bitmaps<<<static_cast<unsigned>(f.cfg.cells), 128>>>(
        d.root_by_edge.get(), f.cfg.edges, f.cfg.cells, d.groups, planes, bitmaps);
    CUDA_CHECK(cudaGetLastError());
}

void direct_map_full(const fixture &f, const device_inputs &d,
                     const std::uint8_t *predicates, std::uint64_t *bitmaps,
                     std::uint8_t *node_scratch) {
    direct_map_full_bitmaps<<<static_cast<unsigned>(f.cfg.cells), 128>>>(
        d.program, f.cfg.cells, predicates, d.root_by_edge.get(), bitmaps, node_scratch);
    CUDA_CHECK(cudaGetLastError());
}

void direct_map_dirty(const fixture &f, const device_inputs &d,
                      const std::uint8_t *predicates, std::uint64_t *bitmaps,
                      std::uint8_t *node_scratch) {
    direct_map_dirty_bitmaps<<<static_cast<unsigned>(f.cfg.cells), 128>>>(
        d.program, f.cfg.cells, predicates, d.dirty_offsets.get(), d.dirty_edges.get(),
        d.root_by_edge.get(), bitmaps, node_scratch);
    CUDA_CHECK(cudaGetLastError());
}

void apply_candidate_dirty_bitmap(const fixture &f, const device_inputs &d,
                                  std::uint32_t *planes, std::uint64_t *bitmaps) {
    update_dirty_bitmaps<<<static_cast<unsigned>(f.cfg.cells), 128>>>(
        d.root_by_edge.get(), d.dirty_offsets.get(), d.dirty_edges.get(), f.cfg.cells,
        f.cfg.edges, d.groups, planes, bitmaps);
    CUDA_CHECK(cudaGetLastError());
}

std::vector<double> measure(std::uint32_t warmup, std::uint32_t repeats,
                            const std::function<void()> &fn,
                            std::vector<double> *wall_samples = nullptr) {
    for (std::uint32_t i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> samples;
    samples.reserve(repeats);
    if (wall_samples) wall_samples->reserve(repeats);
    for (std::uint32_t i = 0; i < repeats; ++i) {
        const auto wall_start = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(start));
        fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
        if (wall_samples) {
            const auto wall_end = std::chrono::steady_clock::now();
            wall_samples->push_back(std::chrono::duration<double, std::milli>(
                wall_end - wall_start).count());
        }
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return samples;
}

struct tripled_samples {
    std::vector<double> control;
    std::vector<double> direct_map;
    std::vector<double> bitplane;
    std::vector<double> control_wall;
    std::vector<double> direct_map_wall;
    std::vector<double> bitplane_wall;
};

tripled_samples measure_tripled(std::uint32_t warmup, std::uint32_t repeats,
                                const std::function<void()> &control,
                                const std::function<void()> &direct_map,
                                const std::function<void()> &bitplane) {
    for (std::uint32_t i = 0; i < warmup; ++i) {
        control();
        direct_map();
        bitplane();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    tripled_samples out;
    out.control.reserve(repeats);
    out.direct_map.reserve(repeats);
    out.bitplane.reserve(repeats);
    out.control_wall.reserve(repeats);
    out.direct_map_wall.reserve(repeats);
    out.bitplane_wall.reserve(repeats);
    std::array<const std::function<void()> *, 3> functions{{
        &control, &direct_map, &bitplane}};
    std::array<std::vector<double> *, 3> device_vectors{{
        &out.control, &out.direct_map, &out.bitplane}};
    std::array<std::vector<double> *, 3> wall_vectors{{
        &out.control_wall, &out.direct_map_wall, &out.bitplane_wall}};
    auto run_one = [&](std::size_t index) {
        const auto wall_start = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(start));
        (*functions[index])();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        device_vectors[index]->push_back(elapsed);
        wall_vectors[index]->push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall_start).count());
    };
    for (std::uint32_t i = 0; i < repeats; ++i) {
        const auto first = static_cast<std::size_t>(i % 3u);
        run_one(first);
        run_one((first + 1u) % 3u);
        run_one((first + 2u) % 3u);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return out;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    if (values.empty()) return 0.0;
    const auto m = values.size() / 2u;
    return values.size() % 2u ? values[m] : 0.5 * (values[m - 1u] + values[m]);
}

double quantile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    if (values.empty()) return 0.0;
    const auto i = static_cast<std::size_t>(std::round(q * (values.size() - 1u)));
    return values[i];
}

std::vector<std::uint64_t> read_bitmap(const device_buffer<std::uint64_t> &bitmap,
                                       const fixture &f) {
    return download(bitmap, f.cfg.cells * ((f.cfg.edges + 63u) / 64u));
}

bool check_worksets(const workset_buffers &w, const fixture &f,
                    const std::vector<std::uint8_t> &predicates) {
    const auto p_offsets = download(w.primal_output_offsets, f.cfg.cells + 1u);
    const auto d_offsets = download(w.derivative_output_offsets, f.cfg.cells + 1u);
    const auto p_items = download(w.primal_output_items, w.primal_output_items.size());
    const auto d_items = download(w.derivative_output_items, w.derivative_output_items.size());
    const auto status = download(w.status, 1u).front();
    if (status.code != static_cast<std::uint32_t>(gate::cuda_workset_code_v1::success)) {
        std::cerr << "workset status=" << status.code << " primal_required="
                  << status.primal_required << " derivative_required="
                  << status.derivative_required << '\n';
        return false;
    }
    std::vector<indexing::indexed_work_item_v1> expected_p, expected_d;
    std::vector<std::uint64_t> expected_po(f.cfg.cells + 1u), expected_do(f.cfg.cells + 1u);
    for (std::uint64_t cell = 0; cell < f.cfg.cells; ++cell) {
        expected_po[cell] = expected_p.size();
        expected_do[cell] = expected_d.size();
        const auto values = scalar_nodes(f, predicates.data() + cell * f.predicate_count);
        std::size_t pi = cell * w.primal_per_cell;
        std::size_t di = cell * w.derivative_per_cell;
        const auto p_end = pi + w.primal_per_cell;
        const auto d_end = di + w.derivative_per_cell;
        while (pi < p_end || di < d_end) {
            const auto p_id = pi < p_end ? f.primal_items[pi].logical_edge_id
                                         : UINT64_MAX;
            const auto d_id = di < d_end ? f.derivative_items[di].logical_edge_id
                                         : UINT64_MAX;
            const auto id = std::min(p_id, d_id);
            const auto edge = id - 1u;
            const bool pass = values[f.root_by_edge[edge]] != 0;
            if (p_id == id) {
                if (pass) expected_p.push_back(f.primal_items[pi]);
                ++pi;
            }
            if (d_id == id) {
                ++di;
            }
            if (pass) {
                if (p_id == id) expected_d.push_back(f.primal_items[pi - 1u]);
                if (d_id == id) expected_d.push_back(f.derivative_items[di - 1u]);
            }
        }
    }
    expected_po[f.cfg.cells] = expected_p.size();
    expected_do[f.cfg.cells] = expected_d.size();
    if (p_offsets != expected_po || d_offsets != expected_do) {
        for (std::uint64_t cell = 0; cell <= f.cfg.cells; ++cell) {
            if (p_offsets[cell] != expected_po[cell] || d_offsets[cell] != expected_do[cell]) {
                std::cerr << "offset mismatch cell=" << cell << " primal=" << p_offsets[cell]
                          << '/' << expected_po[cell] << " derivative=" << d_offsets[cell]
                          << '/' << expected_do[cell] << '\n';
                break;
            }
        }
        return false;
    }
    auto equal_items = [](const auto &expected, const auto &actual) {
        if (expected.size() > actual.size()) return false;
        for (std::size_t i = 0; i < expected.size(); ++i)
            if (expected[i].logical_edge_id != actual[i].logical_edge_id
                || expected[i].logical_edge_ordinal != actual[i].logical_edge_ordinal)
                return false;
        return true;
    };
    if (!equal_items(expected_p, p_items) || !equal_items(expected_d, d_items)) {
        auto report = [](const char *name, const auto &expected, const auto &actual) {
            for (std::size_t i = 0; i < expected.size(); ++i) {
                if (expected[i].logical_edge_id != actual[i].logical_edge_id
                    || expected[i].logical_edge_ordinal != actual[i].logical_edge_ordinal) {
                    std::cerr << name << " item mismatch i=" << i << " actual=("
                              << actual[i].logical_edge_id << ',' << actual[i].logical_edge_ordinal
                              << ") expected=(" << expected[i].logical_edge_id << ','
                              << expected[i].logical_edge_ordinal << ")\n";
                    return;
                }
            }
        };
        report("primal", expected_p, p_items);
        report("derivative", expected_d, d_items);
        std::cerr << "expected counts primal=" << expected_p.size()
                  << " derivative=" << expected_d.size() << '\n';
        return false;
    }
    return true;
}

void require_bitmap_equal(const std::vector<std::uint64_t> &a,
                          const std::vector<std::uint64_t> &b,
                          const char *label) {
    if (a != b) throw std::runtime_error(std::string(label) + " bitmap differs from scalar oracle");
}

void validate_path(const char *name, const device_buffer<std::uint64_t> &bitmap,
                   workset_buffers &worksets, const fixture &f,
                   const std::vector<std::uint8_t> &predicates) {
    CUDA_CHECK(cudaDeviceSynchronize());
    require_bitmap_equal(read_bitmap(bitmap, f), scalar_bitmap(f, predicates), name);
    if (!check_worksets(worksets, f, predicates))
        throw std::runtime_error(std::string(name) + " worksets differ from scalar oracle");
}

void print_samples(const char *key, const std::vector<double> &samples) {
    std::cout << "\"" << key << "\":{\"median_ms\":" << std::setprecision(8)
              << median(samples) << ",\"p90_ms\":" << quantile(samples, 0.90)
              << ",\"samples_ms\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << samples[i];
    }
    std::cout << "]}";
}

void run_one(options cfg) {
    const auto host_begin = std::chrono::steady_clock::now();
    auto f = make_fixture(cfg);
    const auto host_compile_end = std::chrono::steady_clock::now();
    const auto root_compile_begin = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> compiled_roots(cfg.edges, UINT32_MAX);
    for (const auto &binding : f.bindings) {
        if (binding.logical_edge_ordinal >= compiled_roots.size()
            || compiled_roots[binding.logical_edge_ordinal] != UINT32_MAX)
            throw std::runtime_error("invalid or duplicate edge binding in root-map compile");
        compiled_roots[binding.logical_edge_ordinal] = binding.root_node;
    }
    const auto root_compile_end = std::chrono::steady_clock::now();
    const auto device_setup_begin = std::chrono::steady_clock::now();
    device_inputs d(f, compiled_roots);
    workset_buffers w(f);
    const auto words_per_cell = (cfg.edges + 63u) / 64u;
    const auto bitmap_words = cfg.cells * words_per_cell;
    const auto plane_words = f.nodes.size() * d.groups;
    device_buffer<std::uint64_t> baseline_full_bitmap(bitmap_words);
    device_buffer<std::uint64_t> baseline_dirty_bitmap(bitmap_words);
    device_buffer<std::uint64_t> direct_map_full_bitmap(bitmap_words);
    device_buffer<std::uint64_t> direct_map_dirty_bitmap(bitmap_words);
    device_buffer<std::uint64_t> candidate_full_bitmap(bitmap_words);
    device_buffer<std::uint64_t> candidate_dirty_bitmap(bitmap_words);
    device_buffer<std::uint8_t> baseline_node_scratch(cfg.cells * f.nodes.size());
    device_buffer<std::uint32_t> candidate_full_planes(plane_words);
    device_buffer<std::uint32_t> candidate_dirty_planes(plane_words);
    device_buffer<gate::cuda_workset_status_v1> baseline_status(1u);
    const auto device_setup_end = std::chrono::steady_clock::now();

    auto program = d.program;
    const auto input_count = cfg.cells * f.predicate_count;

    // Compare the one-time current bitmap initialization with bitplane
    // compilation/materialization. Both remain device-resident.
    const auto baseline_initialization = measure(0, 5, [&] {
        CUDA_CHECK(gate::cuda_evaluate_condition_bitmaps_v1(program, cfg.cells,
            d.initial_predicates.get(), input_count, baseline_dirty_bitmap.get(), bitmap_words,
            baseline_node_scratch.get(), baseline_node_scratch.size(), baseline_status.get(), nullptr));
    });
    const auto candidate_initialization = measure(0, 5, [&] {
        build_candidate_full_planes(f, d, d.initial_predicates.get(), candidate_dirty_planes.get());
        candidate_materialize(f, d, candidate_dirty_planes.get(), candidate_dirty_bitmap.get(), true);
    });

    // Independent initialization and exact initial-state agreement.
    CUDA_CHECK(gate::cuda_evaluate_condition_bitmaps_v1(program, cfg.cells,
        d.initial_predicates.get(), input_count, baseline_dirty_bitmap.get(), bitmap_words,
        baseline_node_scratch.get(), baseline_node_scratch.size(), baseline_status.get(), nullptr));
    direct_map_full(f, d, d.initial_predicates.get(), direct_map_dirty_bitmap.get(),
                    baseline_node_scratch.get());
    build_candidate_full_planes(f, d, d.initial_predicates.get(), candidate_dirty_planes.get());
    candidate_materialize(f, d, candidate_dirty_planes.get(), candidate_dirty_bitmap.get(), true);
    CUDA_CHECK(cudaDeviceSynchronize());
    require_bitmap_equal(read_bitmap(baseline_dirty_bitmap, f),
                         scalar_bitmap(f, f.initial_predicates), "baseline initial");
    require_bitmap_equal(read_bitmap(direct_map_dirty_bitmap, f),
                         scalar_bitmap(f, f.initial_predicates), "direct-map initial");
    require_bitmap_equal(read_bitmap(candidate_dirty_bitmap, f),
                         scalar_bitmap(f, f.initial_predicates), "bitplane initial");

    // Seed independent full-path outputs with new state before validating.
    CUDA_CHECK(gate::cuda_evaluate_condition_bitmaps_v1(program, cfg.cells,
        d.changed_predicates.get(), input_count, baseline_full_bitmap.get(), bitmap_words,
        baseline_node_scratch.get(), baseline_node_scratch.size(), baseline_status.get(), nullptr));
    w.build(program, f, baseline_full_bitmap.get());
    validate_path("baseline full", baseline_full_bitmap, w, f, f.changed_predicates);

    direct_map_full(f, d, d.changed_predicates.get(), direct_map_full_bitmap.get(),
                    baseline_node_scratch.get());
    w.build(program, f, direct_map_full_bitmap.get());
    validate_path("direct-map full", direct_map_full_bitmap, w, f, f.changed_predicates);

    build_candidate_full_planes(f, d, d.changed_predicates.get(), candidate_full_planes.get());
    candidate_materialize(f, d, candidate_full_planes.get(), candidate_full_bitmap.get(), true);
    w.build(program, f, candidate_full_bitmap.get());
    validate_path("bitplane full", candidate_full_bitmap, w, f, f.changed_predicates);

    // Build and validate the dirty production path from the same old state.
    CUDA_CHECK(gate::cuda_update_condition_bitmaps_v1(program, cfg.cells,
        d.changed_predicates.get(), input_count, d.dirty_offsets.get(), d.dirty_edges.get(),
        f.dirty_edges.size(), baseline_dirty_bitmap.get(), bitmap_words,
        baseline_node_scratch.get(), baseline_node_scratch.size(), baseline_status.get(), nullptr));
    w.build(program, f, baseline_dirty_bitmap.get());
    validate_path("baseline dirty", baseline_dirty_bitmap, w, f, f.changed_predicates);

    direct_map_dirty(f, d, d.changed_predicates.get(), direct_map_dirty_bitmap.get(),
                     baseline_node_scratch.get());
    w.build(program, f, direct_map_dirty_bitmap.get());
    validate_path("direct-map dirty", direct_map_dirty_bitmap, w, f, f.changed_predicates);

    // Candidate update starts from the cached initial planes and bitmap.
    candidate_changed_planes(f, d, d.changed_predicates.get(), candidate_dirty_planes.get());
    apply_candidate_dirty_bitmap(f, d, candidate_dirty_planes.get(), candidate_dirty_bitmap.get());
    w.build(program, f, candidate_dirty_bitmap.get());
    validate_path("bitplane dirty", candidate_dirty_bitmap, w, f, f.changed_predicates);

    if (cfg.check_only) {
        std::cout << "{\"status\":\"correct\",\"edges\":" << cfg.edges
                  << ",\"cells\":" << cfg.cells << ",\"sharing\":\""
                  << cfg.sharing << "\",\"nodes\":" << f.nodes.size()
                  << ",\"dirty_edges_per_cell\":"
                  << (cfg.cells ? f.dirty_edges.size() / cfg.cells : 0u) << "}\n";
        return;
    }

    const auto full_triplet = measure_tripled(2, cfg.repeats, [&] {
        CUDA_CHECK(gate::cuda_evaluate_condition_bitmaps_v1(program, cfg.cells,
            d.changed_predicates.get(), input_count, baseline_full_bitmap.get(), bitmap_words,
            baseline_node_scratch.get(), baseline_node_scratch.size(), baseline_status.get(), nullptr));
        w.build(program, f, baseline_full_bitmap.get());
    }, [&] {
        direct_map_full(f, d, d.changed_predicates.get(), direct_map_full_bitmap.get(),
                        baseline_node_scratch.get());
        w.build(program, f, direct_map_full_bitmap.get());
    }, [&] {
        build_candidate_full_planes(f, d, d.changed_predicates.get(), candidate_full_planes.get());
        candidate_materialize(f, d, candidate_full_planes.get(), candidate_full_bitmap.get(), true);
        w.build(program, f, candidate_full_bitmap.get());
    });
    const auto dirty_triplet = measure_tripled(2, cfg.repeats, [&] {
        CUDA_CHECK(gate::cuda_update_condition_bitmaps_v1(program, cfg.cells,
            d.changed_predicates.get(), input_count, d.dirty_offsets.get(), d.dirty_edges.get(),
            f.dirty_edges.size(), baseline_dirty_bitmap.get(), bitmap_words,
            baseline_node_scratch.get(), baseline_node_scratch.size(), baseline_status.get(), nullptr));
        w.build(program, f, baseline_dirty_bitmap.get());
    }, [&] {
        direct_map_dirty(f, d, d.changed_predicates.get(), direct_map_dirty_bitmap.get(),
                         baseline_node_scratch.get());
        w.build(program, f, direct_map_dirty_bitmap.get());
    }, [&] {
        candidate_changed_planes(f, d, d.changed_predicates.get(), candidate_dirty_planes.get());
        apply_candidate_dirty_bitmap(f, d, candidate_dirty_planes.get(), candidate_dirty_bitmap.get());
        w.build(program, f, candidate_dirty_bitmap.get());
    });
    const auto &full_baseline = full_triplet.control;
    const auto &full_direct_map = full_triplet.direct_map;
    const auto &full_bitplane = full_triplet.bitplane;
    const auto &dirty_baseline = dirty_triplet.control;
    const auto &dirty_direct_map = dirty_triplet.direct_map;
    const auto &dirty_bitplane = dirty_triplet.bitplane;

    const auto current_workset_consumer = measure(2, cfg.repeats, [&] {
        w.build(program, f, baseline_full_bitmap.get());
    });
    const auto candidate_full_plane_build = measure(2, cfg.repeats, [&] {
        build_candidate_full_planes(f, d, d.changed_predicates.get(), candidate_full_planes.get());
    });
    const auto candidate_dirty_plane_update = measure(2, cfg.repeats, [&] {
        candidate_changed_planes(f, d, d.changed_predicates.get(), candidate_dirty_planes.get());
    });
    const auto candidate_full_materialize_and_consume = measure(2, cfg.repeats, [&] {
        candidate_materialize(f, d, candidate_full_planes.get(), candidate_full_bitmap.get(), true);
        w.build(program, f, candidate_full_bitmap.get());
    });
    const auto candidate_dirty_bitmap_and_consume = measure(2, cfg.repeats, [&] {
        apply_candidate_dirty_bitmap(f, d, candidate_dirty_planes.get(), candidate_dirty_bitmap.get());
        w.build(program, f, candidate_dirty_bitmap.get());
    });

    const auto host_end = std::chrono::steady_clock::now();
    const double host_compile_ms = std::chrono::duration<double, std::milli>(
        host_compile_end - host_begin).count();
    const double total_host_ms = std::chrono::duration<double, std::milli>(
        host_end - host_begin).count();
    const double root_compile_ms = std::chrono::duration<double, std::milli>(
        root_compile_end - root_compile_begin).count();
    const double device_setup_ms = std::chrono::duration<double, std::milli>(
        device_setup_end - device_setup_begin).count();
    const auto dirty_per_cell = cfg.cells ? f.dirty_edges.size() / cfg.cells : 0u;
    const auto p_per_cell = w.primal_per_cell;
    const auto d_per_cell = w.derivative_per_cell;
    const double dirty_saving = median(dirty_baseline) - median(dirty_bitplane);
    const double dirty_extra_setup = root_compile_ms + std::max(0.0,
        median(candidate_initialization) - median(baseline_initialization));
    const double break_even = dirty_saving > 0.0 ? dirty_extra_setup / dirty_saving
                                                  : std::numeric_limits<double>::infinity();
    const auto bitmap_bytes = bitmap_words * sizeof(std::uint64_t);
    const auto plane_bytes = plane_words * sizeof(std::uint32_t);
    const auto binding_bytes = f.bindings.size() * sizeof(gate::edge_condition_binding_v1);
    const auto root_map_bytes = f.root_by_edge.size() * sizeof(std::uint32_t);
    const auto node_scratch_bytes = cfg.cells * f.nodes.size() * sizeof(std::uint8_t);

    std::cout << "{\"status\":\"measured\",\"edges\":" << cfg.edges
              << ",\"cells\":" << cfg.cells << ",\"sharing\":\"" << cfg.sharing
              << "\",\"predicates\":" << f.predicate_count
              << ",\"seed\":" << cfg.seed
              << ",\"nodes\":" << f.nodes.size()
              << ",\"bound_edges\":" << f.bindings.size()
              << ",\"dirty_edges_per_cell\":" << dirty_per_cell
              << ",\"primal_candidates_per_cell\":" << p_per_cell
              << ",\"derivative_candidates_per_cell\":" << d_per_cell
              << ",\"plane_extra_bytes\":" << (plane_bytes + root_map_bytes)
              << ",\"direct_map_extra_bytes\":" << root_map_bytes
              << ",\"baseline_bitmap_bytes\":" << bitmap_bytes
              << ",\"baseline_node_scratch_bytes\":" << node_scratch_bytes
              << ",\"baseline_binding_bytes\":" << binding_bytes
              << ",\"candidate_plane_bytes\":" << plane_bytes
              << ",\"candidate_edge_root_map_bytes\":" << root_map_bytes
              << ",\"host_fixture_compile_ms\":" << host_compile_ms
              << ",\"candidate_edge_root_compile_ms\":" << root_compile_ms
              << ",\"shared_host_device_setup_ms\":" << device_setup_ms
              << ",\"host_setup_through_run_ms\":" << total_host_ms
              << ",\"timing_clock\":\"cuda_events_plus_host_wall\""
              << ",\"dirty_savings_ms\":" << dirty_saving
              << ",\"dirty_reuse_break_even_updates\":";
    if (std::isfinite(break_even)) std::cout << break_even;
    else std::cout << "null";
    std::cout << ',';
    print_samples("current_full_bitmap_plus_worksets", full_baseline);
    std::cout << ',';
    print_samples("direct_root_map_scalar_full_plus_worksets", full_direct_map);
    std::cout << ',';
    print_samples("bitplane_full_plus_materialize_plus_worksets", full_bitplane);
    std::cout << ',';
    print_samples("current_dirty_bitmap_plus_worksets", dirty_baseline);
    std::cout << ',';
    print_samples("direct_root_map_scalar_dirty_plus_worksets", dirty_direct_map);
    std::cout << ',';
    print_samples("bitplane_incremental_dirty_plus_worksets", dirty_bitplane);
    std::cout << ',';
    print_samples("current_full_host_wall", full_triplet.control_wall);
    std::cout << ',';
    print_samples("direct_root_map_full_host_wall", full_triplet.direct_map_wall);
    std::cout << ',';
    print_samples("bitplane_full_host_wall", full_triplet.bitplane_wall);
    std::cout << ',';
    print_samples("current_dirty_host_wall", dirty_triplet.control_wall);
    std::cout << ',';
    print_samples("direct_root_map_dirty_host_wall", dirty_triplet.direct_map_wall);
    std::cout << ',';
    print_samples("bitplane_dirty_host_wall", dirty_triplet.bitplane_wall);
    std::cout << ',';
    print_samples("current_initial_bitmap_only", baseline_initialization);
    std::cout << ',';
    print_samples("bitplane_initial_transpose_eval_materialize", candidate_initialization);
    std::cout << ',';
    print_samples("current_workset_consumer_only", current_workset_consumer);
    std::cout << ',';
    print_samples("bitplane_full_plane_build_only", candidate_full_plane_build);
    std::cout << ',';
    print_samples("bitplane_dirty_plane_update_only", candidate_dirty_plane_update);
    std::cout << ',';
    print_samples("bitplane_full_materialize_plus_consumer", candidate_full_materialize_and_consume);
    std::cout << ',';
    print_samples("bitplane_dirty_bitmap_plus_consumer", candidate_dirty_bitmap_and_consume);
    std::cout << "}\n";
}

options parse(int argc, char **argv) {
    options cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char *name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--edges") cfg.edges = std::stoull(value("--edges"));
        else if (arg == "--cells") cfg.cells = std::stoull(value("--cells"));
        else if (arg == "--seed") cfg.seed = std::stoull(value("--seed"));
        else if (arg == "--repeats") cfg.repeats = static_cast<std::uint32_t>(std::stoul(value("--repeats")));
        else if (arg == "--sharing") cfg.sharing = value("--sharing");
        else if (arg == "--check-only") cfg.check_only = true;
        else throw std::runtime_error("unknown option: " + arg);
    }
    if (cfg.edges == 0 || cfg.cells == 0 || cfg.repeats == 0)
        throw std::runtime_error("edges, cells, and repeats must be positive");
    return cfg;
}

} // namespace

int main(int argc, char **argv) {
    try {
        bool suite = false;
        bool suite_check_only = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--suite") suite = true;
            if (std::string(argv[i]) == "--suite-check-only") {
                suite = true;
                suite_check_only = true;
            }
        }
        if (!suite) {
            run_one(parse(argc, argv));
            return 0;
        }
        const auto repeats = [&] {
            for (int i = 1; i + 1 < argc; ++i)
                if (std::string(argv[i]) == "--repeats") return static_cast<std::uint32_t>(std::stoul(argv[i + 1]));
            return 11u;
        }();
        for (const auto &shape : std::array<std::tuple<std::uint64_t,std::uint64_t,const char *>,4>{
                 std::tuple<std::uint64_t,std::uint64_t,const char *>{2176,64,"few"},
                 {2176,512,"few"}, {65536,64,"few"}, {2176,64,"weak"}}) {
            options cfg;
            cfg.edges = std::get<0>(shape);
            cfg.cells = std::get<1>(shape);
            cfg.sharing = std::get<2>(shape);
            cfg.repeats = repeats;
            cfg.check_only = suite_check_only;
            run_one(cfg);
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "shared_condition_probe: " << e.what() << '\n';
        return 2;
    }
}
