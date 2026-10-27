#include <Cellerator/compute/indexing/cuda_postings.cuh>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace ix = cellerator::compute::indexing;
using clock_type = std::chrono::steady_clock;

#define CUDA_CHECK(call) do { \
    const cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) throw std::runtime_error( \
        std::string(#call) + ": " + cudaGetErrorString(err_)); \
} while (0)

namespace {

constexpr std::uint64_t kSeed = 20261027;
constexpr std::uint64_t kStructureEpoch = 1;
constexpr std::uint64_t kRelationId = 7001;
constexpr std::uint64_t kStructureId = 9001;

struct config {
    std::string id;
    std::uint64_t edges;
    std::uint64_t cells;
    std::uint64_t entities = 256;
    double skew = 0.0;
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

struct edge {
    std::uint64_t source;
    std::uint64_t target;
    std::uint64_t cofactor;
};

struct posting_record {
    std::uint32_t role;
    std::uint64_t entity;
    std::uint64_t logical_id;
    std::uint64_t ordinal;
};

struct fixture {
    std::vector<edge> edges;
    std::uint64_t digest = 1469598103934665603ull;
};

std::uint64_t choose_source(std::uint64_t &rng, const config &cfg) {
    if (unit_random(rng) < cfg.skew) return 1;
    return 1 + random_next(rng) % cfg.entities;
}

fixture make_fixture(const config &cfg) {
    fixture f;
    f.edges.resize(cfg.edges);
    const auto delta_edges = cfg.edges > 1
        ? std::max<std::uint64_t>(1, cfg.edges / 16) : 0;
    const auto base_edges = cfg.edges - delta_edges;
    std::uint64_t rng = kSeed;
    auto fill = [&](std::uint64_t begin, std::uint64_t count) {
        for (std::uint64_t local = 0; local < count; ++local) {
            const auto ordinal = begin + local;
            const auto source = choose_source(rng, cfg);
            const auto target = 1 + random_next(rng) % cfg.entities;
            const auto cofactor = 1 + random_next(rng) % cfg.entities;
            f.edges[ordinal] = {source, target, cofactor};
            const std::uint32_t participant_count = ordinal % 31 == 0 ? 4u : 3u;
            f.digest = mix_hash(f.digest, ordinal + 1);
            f.digest = mix_hash(f.digest, source);
            f.digest = mix_hash(f.digest, target);
            f.digest = mix_hash(f.digest, cofactor);
            f.digest = mix_hash(f.digest, participant_count);
        }
    };
    fill(0, base_edges);
    fill(base_edges, delta_edges);
    return f;
}

struct query_set {
    std::vector<std::uint64_t> offsets;
    std::vector<ix::posting_term_v1> terms;
};

query_set make_queries(const config &cfg) {
    query_set q;
    const std::uint64_t rows = 2 * cfg.cells;
    q.offsets.resize(rows + 1);
    q.terms.reserve(rows);
    std::uint64_t rng = kSeed ^ 0xd1b54a32d192ed03ull;
    auto choose_query = [&](std::uint32_t role, std::uint64_t cell) {
        const bool hot = role == 0 && cfg.skew > 0.5 && cell % 4 == 0;
        return hot ? 1 : 1 + random_next(rng) % cfg.entities;
    };
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        q.offsets[2 * cell] = 2 * cell;
        q.terms.push_back({0, 0, choose_query(0, cell)});
        q.offsets[2 * cell + 1] = 2 * cell + 1;
        q.terms.push_back({1, 0, choose_query(1, cell)});
    }
    q.offsets[rows] = rows;
    return q;
}

std::uint64_t combined_fixture_digest(const config &cfg,
    const fixture &f, const query_set &q) {
    std::uint64_t in = 1469598103934665603ull;
    in = mix_hash(in, cfg.edges);
    in = mix_hash(in, cfg.cells);
    in = mix_hash(in, cfg.entities);
    in = mix_hash(in, kSeed);
    const auto dirty_per_cell = std::max<std::uint64_t>(1,
        static_cast<std::uint64_t>(std::ceil(cfg.edges * 0.01)));
    for (std::uint64_t edge = 0; edge < cfg.edges; ++edge) {
        std::uint32_t root = 0;
        if (edge < dirty_per_cell)
            root = edge % 3 == 0 ? 0 : edge % 3 == 1 ? 4 : 6;
        else {
            const std::array<std::uint32_t, 4> roots{{1, 2, 3, 5}};
            root = roots[(edge - dirty_per_cell) % roots.size()];
        }
        in = mix_hash(in, edge);
        in = mix_hash(in, root);
    }
    std::uint64_t rng = kSeed ^ 0x94d049bb133111ebull;
    for (std::uint64_t i = 0; i < cfg.cells * 4; ++i)
        in = mix_hash(in, unit_random(rng) < 0.5 ? 1u : 0u);
    in = mix_hash(in, dirty_per_cell);
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell)
        for (std::uint64_t ordinal = 0; ordinal < dirty_per_cell; ++ordinal)
            in = mix_hash(in, ordinal);
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        in = mix_hash(in, q.terms[2 * cell].role);
        in = mix_hash(in, q.terms[2 * cell].entity_id);
    }
    for (std::uint64_t cell = 0; cell < cfg.cells; ++cell) {
        in = mix_hash(in, q.terms[2 * cell + 1].role);
        in = mix_hash(in, q.terms[2 * cell + 1].entity_id);
    }
    return mix_hash(f.digest, in);
}

struct posting_projection {
    std::vector<ix::posting_term_v1> terms;
    std::vector<std::uint64_t> offsets;
    std::vector<ix::indexed_work_item_v1> items;
    std::uint64_t raw_incidence_count = 0;
};

posting_projection make_postings(const fixture &f) {
    std::vector<posting_record> records;
    records.reserve(f.edges.size() * 4);
    for (std::uint64_t ordinal = 0; ordinal < f.edges.size(); ++ordinal) {
        const auto &e = f.edges[ordinal];
        const auto id = ordinal + 1;
        records.push_back({0, e.source, id, ordinal});
        records.push_back({1, e.target, id, ordinal});
        records.push_back({2, e.cofactor, id, ordinal});
        if (ordinal % 31 == 0)
            records.push_back({0, e.source, id, ordinal});
    }
    posting_projection p;
    p.raw_incidence_count = records.size();
    auto key = [](const posting_record &x) {
        return std::tie(x.role, x.entity, x.logical_id);
    };
    std::sort(records.begin(), records.end(), [&](const auto &a, const auto &b) {
        return key(a) < key(b);
    });
    records.erase(std::unique(records.begin(), records.end(), [&](const auto &a, const auto &b) {
        return key(a) == key(b);
    }), records.end());
    p.offsets.push_back(0);
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto &r = records[i];
        if (i == 0 || records[i - 1].role != r.role
            || records[i - 1].entity != r.entity) {
            if (i != 0) p.offsets.push_back(p.items.size());
            p.terms.push_back({r.role, 0, r.entity});
        }
        p.items.push_back({r.logical_id, r.ordinal});
    }
    p.offsets.push_back(p.items.size());
    return p;
}

struct bitmap_projection {
    std::vector<std::uint32_t> words;
    std::uint64_t row_count = 0;
    std::uint64_t words_per_row = 0;
};

bitmap_projection make_bitmaps(const fixture &f, const config &cfg) {
    bitmap_projection b;
    b.row_count = 3 * cfg.entities;
    b.words_per_row = (cfg.edges + 31) / 32;
    if (b.row_count > std::numeric_limits<std::uint64_t>::max() / b.words_per_row
        || b.row_count * b.words_per_row > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("dense term bitmap exceeds host addressable storage");
    b.words.assign(static_cast<std::size_t>(b.row_count * b.words_per_row), 0u);
    auto set = [&](std::uint32_t role, std::uint64_t entity, std::uint64_t ordinal) {
        const auto row = static_cast<std::uint64_t>(role) * cfg.entities + entity - 1;
        b.words[row * b.words_per_row + ordinal / 32] |= 1u << (ordinal % 32);
    };
    for (std::uint64_t ordinal = 0; ordinal < f.edges.size(); ++ordinal) {
        const auto &e = f.edges[ordinal];
        set(0, e.source, ordinal);
        set(1, e.target, ordinal);
        set(2, e.cofactor, ordinal);
    }
    return b;
}

std::vector<std::vector<ix::indexed_work_item_v1>> scalar_oracle(
    const fixture &f, const query_set &q) {
    const auto rows = q.offsets.size() - 1;
    std::vector<std::vector<ix::indexed_work_item_v1>> out(rows);
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t ordinal = 0; ordinal < f.edges.size(); ++ordinal) {
            const auto &e = f.edges[ordinal];
            bool eligible = false;
            for (auto i = q.offsets[row]; i < q.offsets[row + 1]; ++i) {
                const auto &t = q.terms[i];
                const auto entity = t.role == 0 ? e.source
                    : t.role == 1 ? e.target : t.role == 2 ? e.cofactor : 0;
                eligible = eligible || entity == t.entity_id;
            }
            if (eligible) out[row].push_back({ordinal + 1, ordinal});
        }
    }
    return out;
}

template<class T> struct device_buffer {
    T *ptr = nullptr;
    std::size_t size = 0;
    device_buffer() = default;
    explicit device_buffer(std::size_t n) : size(n) {
        if (n) CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)));
    }
    ~device_buffer() { if (ptr) cudaFree(ptr); }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    device_buffer(device_buffer &&o) noexcept : ptr(o.ptr), size(o.size) {
        o.ptr = nullptr; o.size = 0;
    }
    device_buffer &operator=(device_buffer &&o) noexcept {
        if (this != &o) {
            if (ptr) cudaFree(ptr);
            ptr = o.ptr; size = o.size; o.ptr = nullptr; o.size = 0;
        }
        return *this;
    }
    std::uint64_t bytes() const { return static_cast<std::uint64_t>(size) * sizeof(T); }
};

struct candidate_status {
    std::uint32_t code;
    std::uint32_t reserved;
    std::uint64_t required;
};

__global__ void bitmap_union_count_kernel(
    const std::uint64_t *query_offsets, const ix::posting_term_v1 *queries,
    std::uint64_t query_count, std::uint64_t rows, std::uint64_t entities,
    std::uint64_t edge_count, std::uint64_t words_per_row,
    const std::uint32_t *term_bitmaps, std::uint32_t *union_words,
    std::uint32_t *word_counts, candidate_status *status) {
    const std::uint64_t word_index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const std::uint64_t total_words = rows * words_per_row;
    if (word_index >= total_words) return;
    const auto row = word_index / words_per_row;
    const auto word = word_index - row * words_per_row;
    const auto begin = query_offsets[row];
    const auto end = query_offsets[row + 1];
    if (begin > end || end > query_count) {
        atomicCAS(&status->code, 0u, 2u);
        union_words[word_index] = 0u;
        word_counts[word_index] = 0u;
        return;
    }
    std::uint32_t bits = 0;
    for (auto i = begin; i < end; ++i) {
        const auto term = queries[i];
        if (term.reserved != 0 || term.role >= 3 || term.entity_id == 0
            || term.entity_id > entities) {
            atomicCAS(&status->code, 0u, 2u);
            continue;
        }
        const std::uint64_t bitmap_row = term.role * entities + term.entity_id - 1;
        bits |= term_bitmaps[bitmap_row * words_per_row + word];
    }
    const std::uint64_t first_ordinal = word * 32;
    if (first_ordinal + 32 > edge_count && edge_count % 32 != 0)
        bits &= (1u << (edge_count % 32)) - 1u;
    union_words[word_index] = bits;
    word_counts[word_index] = __popc(bits);
}

__global__ void bitmap_finish_scan_kernel(
    std::uint64_t rows, std::uint64_t words_per_row,
    const std::uint32_t *word_counts, const std::uint64_t *word_prefix,
    std::uint64_t output_capacity, std::uint64_t *output_offsets,
    candidate_status *status) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || status->code != 0) return;
    const auto last_word = rows * words_per_row - 1;
    const auto required = word_prefix[last_word] + word_counts[last_word];
    status->required = required;
    if (required > output_capacity) {
        status->code = 1u;
        return;
    }
    output_offsets[0] = 0;
    for (std::uint64_t row = 0; row < rows; ++row) {
        const auto first = row * words_per_row;
        const auto last = first + words_per_row - 1;
        output_offsets[row + 1] = word_prefix[last] + word_counts[last];
    }
}

__global__ void bitmap_emit_kernel(
    std::uint64_t rows, std::uint64_t edge_count,
    std::uint64_t words_per_row, const std::uint32_t *union_words,
    const std::uint64_t *word_prefix, const std::uint64_t *logical_ids,
    ix::indexed_work_item_v1 *output, candidate_status *status) {
    const std::uint64_t word_index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (word_index >= rows * words_per_row || status->code != 0) return;
    std::uint32_t bits = union_words[word_index];
    std::uint64_t output_index = word_prefix[word_index];
    const auto first_ordinal = (word_index % words_per_row) * 32;
    while (bits) {
        const int bit = __ffs(bits) - 1;
        const auto ordinal = first_ordinal + static_cast<std::uint64_t>(bit);
        if (ordinal < edge_count)
            output[output_index++] = {logical_ids[ordinal], ordinal};
        bits &= bits - 1;
    }
}

constexpr unsigned bitmap_block_threads = 128;

__global__ void bitmap_row_build_kernel(
    const std::uint64_t *query_offsets, const ix::posting_term_v1 *queries,
    std::uint64_t query_count, std::uint64_t rows, std::uint64_t entities,
    std::uint64_t edge_count, std::uint64_t words_per_row,
    const std::uint32_t *term_bitmaps, std::uint32_t *union_words,
    std::uint32_t *word_counts, std::uint64_t *row_counts,
    candidate_status *status) {
    using block_reduce = cub::BlockReduce<std::uint64_t, bitmap_block_threads>;
    __shared__ typename block_reduce::TempStorage temp;
    const auto row = static_cast<std::uint64_t>(blockIdx.x);
    if (row >= rows) return;
    const auto begin = query_offsets[row];
    const auto end = query_offsets[row + 1];
    if (begin > end || end > query_count) {
        if (threadIdx.x == 0) atomicCAS(&status->code, 0u, 2u);
        if (threadIdx.x == 0) row_counts[row] = 0;
        return;
    }
    std::uint64_t local_count = 0;
    for (std::uint64_t word = threadIdx.x; word < words_per_row;
         word += blockDim.x) {
        std::uint32_t bits = 0;
        for (auto i = begin; i < end; ++i) {
            const auto term = queries[i];
            if (term.reserved != 0 || term.role >= 3 || term.entity_id == 0
                || term.entity_id > entities) {
                atomicCAS(&status->code, 0u, 2u);
                continue;
            }
            const std::uint64_t bitmap_row = term.role * entities + term.entity_id - 1;
            bits |= term_bitmaps[bitmap_row * words_per_row + word];
        }
        const auto first_ordinal = word * 32;
        if (first_ordinal + 32 > edge_count && edge_count % 32 != 0)
            bits &= (1u << (edge_count % 32)) - 1u;
        const auto index = row * words_per_row + word;
        union_words[index] = bits;
        const auto count = static_cast<std::uint32_t>(__popc(bits));
        word_counts[index] = count;
        local_count += count;
    }
    const auto total = block_reduce(temp).Sum(local_count);
    if (threadIdx.x == 0) row_counts[row] = total;
}

template<int ItemsPerThread>
__global__ void bitmap_row_compact_kernel(
    std::uint64_t rows, std::uint64_t edge_count,
    std::uint64_t words_per_row, std::uint64_t output_capacity,
    const std::uint32_t *union_words, const std::uint32_t *word_counts,
    const std::uint64_t *row_counts, const std::uint64_t *logical_ids,
    std::uint64_t *output_offsets, ix::indexed_work_item_v1 *output,
    candidate_status *status) {
    using block_scan = cub::BlockScan<std::uint32_t, bitmap_block_threads,
        cub::BLOCK_SCAN_RAKING>;
    __shared__ typename block_scan::TempStorage temp;
    __shared__ std::uint64_t shared_row_base;
    __shared__ std::uint64_t shared_required;
    const auto row = static_cast<std::uint64_t>(blockIdx.x);
    if (row >= rows) return;

    if (threadIdx.x == 0) {
        std::uint64_t row_base = 0;
        std::uint64_t required = 0;
        for (std::uint64_t i = 0; i < rows; ++i) {
            const auto count = row_counts[i];
            if (i < row) row_base += count;
            required += count;
        }
        shared_row_base = row_base;
        shared_required = required;
        if (row == 0) {
            status->required = required;
            if (required > output_capacity) status->code = 1u;
            if (required <= output_capacity && status->code == 0u)
                output_offsets[0] = 0;
        }
    }
    __syncthreads();
    if (shared_required > output_capacity || status->code != 0u) return;
    if (threadIdx.x == 0)
        output_offsets[row + 1] = shared_row_base + row_counts[row];

    std::uint32_t item_counts[ItemsPerThread];
    std::uint32_t item_prefix[ItemsPerThread];
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const auto word = static_cast<std::uint64_t>(threadIdx.x)
            * ItemsPerThread + item;
        item_counts[item] = word < words_per_row
            ? word_counts[row * words_per_row + word] : 0u;
    }
    block_scan(temp).ExclusiveSum(item_counts, item_prefix);
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const auto word = static_cast<std::uint64_t>(threadIdx.x)
            * ItemsPerThread + item;
        if (word >= words_per_row) continue;
        std::uint32_t bits = union_words[row * words_per_row + word];
        std::uint64_t output_index = shared_row_base + item_prefix[item];
        const auto first_ordinal = word * 32;
        while (bits) {
            const int bit = __ffs(bits) - 1;
            const auto ordinal = first_ordinal + static_cast<std::uint64_t>(bit);
            if (ordinal < edge_count)
                output[output_index++] = {logical_ids[ordinal], ordinal};
            bits &= bits - 1;
        }
    }
}

struct common_device {
    device_buffer<std::uint64_t> query_offsets;
    device_buffer<ix::posting_term_v1> query_terms;
    std::uint64_t memory_bytes = 0;
    double allocation_ms = 0;
    double h2d_ms = 0;
};

struct baseline_device {
    device_buffer<ix::posting_term_v1> terms;
    device_buffer<std::uint64_t> term_offsets;
    device_buffer<ix::indexed_work_item_v1> items;
    device_buffer<ix::cuda_posting_scratch_v1::cursor_v1> cursors;
    device_buffer<std::uint64_t> counts;
    device_buffer<ix::cuda_posting_status_v1> status;
    device_buffer<std::uint64_t> output_offsets;
    device_buffer<ix::indexed_work_item_v1> output_items;
    ix::cuda_reverse_posting_view_v1 view{};
    ix::cuda_posting_scratch_v1 scratch{};
    ix::cuda_posting_output_v1 output{};
    std::uint64_t index_bytes = 0;
    std::uint64_t workspace_output_bytes = 0;
    double allocation_ms = 0;
    double h2d_ms = 0;
};

struct bitmap_device {
    device_buffer<std::uint32_t> term_bitmaps;
    device_buffer<std::uint64_t> edge_ids;
    device_buffer<std::uint32_t> union_words;
    device_buffer<std::uint32_t> word_counts;
    device_buffer<std::uint64_t> word_prefix;
    device_buffer<std::uint64_t> output_offsets;
    device_buffer<ix::indexed_work_item_v1> output_items;
    device_buffer<candidate_status> status;
    device_buffer<std::uint8_t> scan_scratch;
    std::size_t scan_scratch_bytes = 0;
    std::uint64_t output_capacity = 0;
    std::uint64_t rows = 0;
    std::uint64_t words_per_row = 0;
    std::uint64_t edges = 0;
    std::uint64_t entities = 0;
    std::uint64_t index_bytes = 0;
    std::uint64_t workspace_output_bytes = 0;
    double allocation_ms = 0;
    double index_allocation_ms = 0;
    double workspace_allocation_ms = 0;
    double h2d_ms = 0;
};

struct bitmap_two_kernel_device {
    device_buffer<std::uint32_t> union_words;
    device_buffer<std::uint32_t> word_counts;
    device_buffer<std::uint64_t> row_counts;
    device_buffer<std::uint64_t> output_offsets;
    device_buffer<ix::indexed_work_item_v1> output_items;
    device_buffer<candidate_status> status;
    std::uint64_t output_capacity = 0;
    std::uint64_t rows = 0;
    std::uint64_t words_per_row = 0;
    std::uint64_t edges = 0;
    std::uint64_t entities = 0;
    std::uint64_t index_bytes = 0;
    std::uint64_t workspace_output_bytes = 0;
    double allocation_ms = 0;
};

double event_elapsed(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

template<class F> double measure_copy_batch(F enqueue) {
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    enqueue();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const auto ms = event_elapsed(start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

common_device prepare_common(const query_set &q) {
    common_device d;
    const auto alloc_begin = clock_type::now();
    d.query_offsets = device_buffer<std::uint64_t>(q.offsets.size());
    d.query_terms = device_buffer<ix::posting_term_v1>(q.terms.size());
    d.allocation_ms = std::chrono::duration<double, std::milli>(clock_type::now() - alloc_begin).count();
    d.memory_bytes = d.query_offsets.bytes() + d.query_terms.bytes();
    d.h2d_ms = measure_copy_batch([&] {
        if (d.query_offsets.bytes()) CUDA_CHECK(cudaMemcpyAsync(d.query_offsets.ptr,
            q.offsets.data(), d.query_offsets.bytes(), cudaMemcpyHostToDevice));
        if (d.query_terms.bytes()) CUDA_CHECK(cudaMemcpyAsync(d.query_terms.ptr,
            q.terms.data(), d.query_terms.bytes(), cudaMemcpyHostToDevice));
    });
    return d;
}

baseline_device prepare_baseline(const posting_projection &p,
    const common_device &common, std::uint64_t rows,
    std::uint64_t query_count, std::uint64_t edge_count) {
    baseline_device d;
    const auto capacity = rows * edge_count;
    const auto alloc_begin = clock_type::now();
    d.terms = device_buffer<ix::posting_term_v1>(p.terms.size());
    d.term_offsets = device_buffer<std::uint64_t>(p.offsets.size());
    d.items = device_buffer<ix::indexed_work_item_v1>(p.items.size());
    d.cursors = device_buffer<ix::cuda_posting_scratch_v1::cursor_v1>(query_count);
    d.counts = device_buffer<std::uint64_t>(rows);
    d.status = device_buffer<ix::cuda_posting_status_v1>(1);
    d.output_offsets = device_buffer<std::uint64_t>(rows + 1);
    d.output_items = device_buffer<ix::indexed_work_item_v1>(capacity);
    d.allocation_ms = std::chrono::duration<double, std::milli>(clock_type::now() - alloc_begin).count();
    d.index_bytes = d.terms.bytes() + d.term_offsets.bytes() + d.items.bytes();
    d.workspace_output_bytes = d.cursors.bytes() + d.counts.bytes() + d.status.bytes()
        + d.output_offsets.bytes() + d.output_items.bytes();
    d.h2d_ms = measure_copy_batch([&] {
        if (d.terms.bytes()) CUDA_CHECK(cudaMemcpyAsync(d.terms.ptr,
            p.terms.data(), d.terms.bytes(), cudaMemcpyHostToDevice));
        if (d.term_offsets.bytes()) CUDA_CHECK(cudaMemcpyAsync(d.term_offsets.ptr,
            p.offsets.data(), d.term_offsets.bytes(), cudaMemcpyHostToDevice));
        if (d.items.bytes()) CUDA_CHECK(cudaMemcpyAsync(d.items.ptr,
            p.items.data(), d.items.bytes(), cudaMemcpyHostToDevice));
    });
    d.view = {kRelationId, kStructureId, kStructureEpoch, d.terms.ptr,
        d.term_offsets.ptr, p.terms.size(), d.items.ptr, p.items.size()};
    d.scratch = {d.cursors.ptr, query_count, d.counts.ptr, d.status.ptr};
    d.output = {d.output_offsets.ptr, rows + 1, d.output_items.ptr, capacity};
    (void)common;
    return d;
}

bitmap_device prepare_bitmap(const bitmap_projection &b,
    const fixture &f, const common_device &common, const config &cfg,
    std::uint64_t rows) {
    bitmap_device d;
    d.rows = rows;
    d.words_per_row = b.words_per_row;
    d.edges = cfg.edges;
    d.entities = cfg.entities;
    d.output_capacity = rows * cfg.edges;
    const auto total_words = rows * b.words_per_row;
    const auto index_alloc_begin = clock_type::now();
    d.term_bitmaps = device_buffer<std::uint32_t>(b.words.size());
    d.edge_ids = device_buffer<std::uint64_t>(cfg.edges);
    d.index_allocation_ms = std::chrono::duration<double, std::milli>(
        clock_type::now() - index_alloc_begin).count();
    const auto workspace_alloc_begin = clock_type::now();
    d.union_words = device_buffer<std::uint32_t>(total_words);
    d.word_counts = device_buffer<std::uint32_t>(total_words);
    d.word_prefix = device_buffer<std::uint64_t>(total_words);
    d.output_offsets = device_buffer<std::uint64_t>(rows + 1);
    d.output_items = device_buffer<ix::indexed_work_item_v1>(d.output_capacity);
    d.status = device_buffer<candidate_status>(1);
    std::size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes,
        d.word_counts.ptr, d.word_prefix.ptr, total_words));
    d.scan_scratch_bytes = temp_bytes;
    d.scan_scratch = device_buffer<std::uint8_t>(temp_bytes);
    d.workspace_allocation_ms = std::chrono::duration<double, std::milli>(
        clock_type::now() - workspace_alloc_begin).count();
    d.allocation_ms = d.index_allocation_ms + d.workspace_allocation_ms;
    d.index_bytes = d.term_bitmaps.bytes() + d.edge_ids.bytes();
    d.workspace_output_bytes = d.union_words.bytes() + d.word_counts.bytes()
        + d.word_prefix.bytes() + d.output_offsets.bytes() + d.output_items.bytes()
        + d.status.bytes() + d.scan_scratch.bytes();
    std::vector<std::uint64_t> ids(cfg.edges);
    std::iota(ids.begin(), ids.end(), 1ull);
    d.h2d_ms = measure_copy_batch([&] {
        CUDA_CHECK(cudaMemcpyAsync(d.term_bitmaps.ptr, b.words.data(),
            d.term_bitmaps.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d.edge_ids.ptr, ids.data(),
            d.edge_ids.bytes(), cudaMemcpyHostToDevice));
    });
    (void)common;
    return d;
}

bitmap_two_kernel_device prepare_bitmap_two_kernel(
    const bitmap_device &shared_index, const config &cfg, std::uint64_t rows) {
    bitmap_two_kernel_device d;
    d.rows = rows;
    d.words_per_row = shared_index.words_per_row;
    d.edges = cfg.edges;
    d.entities = cfg.entities;
    d.output_capacity = rows * cfg.edges;
    d.index_bytes = shared_index.index_bytes;
    const auto total_words = rows * d.words_per_row;
    const auto alloc_begin = clock_type::now();
    d.union_words = device_buffer<std::uint32_t>(total_words);
    d.word_counts = device_buffer<std::uint32_t>(total_words);
    d.row_counts = device_buffer<std::uint64_t>(rows);
    d.output_offsets = device_buffer<std::uint64_t>(rows + 1);
    d.output_items = device_buffer<ix::indexed_work_item_v1>(d.output_capacity);
    d.status = device_buffer<candidate_status>(1);
    d.allocation_ms = std::chrono::duration<double, std::milli>(
        clock_type::now() - alloc_begin).count();
    d.workspace_output_bytes = d.union_words.bytes() + d.word_counts.bytes()
        + d.row_counts.bytes() + d.output_offsets.bytes()
        + d.output_items.bytes() + d.status.bytes();
    return d;
}

void launch_bitmap(bitmap_device &d, const common_device &c,
    const query_set &q, std::uint64_t output_capacity, cudaStream_t stream) {
    const auto rows = d.rows;
    const auto words = rows * d.words_per_row;
    CUDA_CHECK(cudaMemsetAsync(d.status.ptr, 0, sizeof(candidate_status), stream));
    constexpr unsigned threads = 256;
    bitmap_union_count_kernel<<<static_cast<unsigned>((words + threads - 1) / threads),
        threads, 0, stream>>>(c.query_offsets.ptr, c.query_terms.ptr, q.terms.size(),
        rows, d.entities, d.edges, d.words_per_row, d.term_bitmaps.ptr,
        d.union_words.ptr, d.word_counts.ptr, d.status.ptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d.scan_scratch.ptr, d.scan_scratch_bytes,
        d.word_counts.ptr, d.word_prefix.ptr, words, stream));
    bitmap_finish_scan_kernel<<<1, 1, 0, stream>>>(rows, d.words_per_row,
        d.word_counts.ptr, d.word_prefix.ptr, output_capacity,
        d.output_offsets.ptr, d.status.ptr);
    CUDA_CHECK(cudaGetLastError());
    bitmap_emit_kernel<<<static_cast<unsigned>((words + threads - 1) / threads),
        threads, 0, stream>>>(rows, d.edges, d.words_per_row, d.union_words.ptr,
        d.word_prefix.ptr, d.edge_ids.ptr, d.output_items.ptr, d.status.ptr);
    CUDA_CHECK(cudaGetLastError());
}

void launch_bitmap_two_kernel(bitmap_two_kernel_device &d,
    const bitmap_device &index, const common_device &c, const query_set &q,
    std::uint64_t output_capacity, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(d.status.ptr, 0, sizeof(candidate_status), stream));
    bitmap_row_build_kernel<<<static_cast<unsigned>(d.rows), bitmap_block_threads,
        0, stream>>>(c.query_offsets.ptr, c.query_terms.ptr, q.terms.size(),
        d.rows, d.entities, d.edges, d.words_per_row, index.term_bitmaps.ptr,
        d.union_words.ptr, d.word_counts.ptr, d.row_counts.ptr, d.status.ptr);
    CUDA_CHECK(cudaGetLastError());
    if (d.words_per_row <= bitmap_block_threads) {
        bitmap_row_compact_kernel<1><<<static_cast<unsigned>(d.rows),
            bitmap_block_threads, 0, stream>>>(d.rows, d.edges,
            d.words_per_row, output_capacity, d.union_words.ptr,
            d.word_counts.ptr, d.row_counts.ptr, index.edge_ids.ptr,
            d.output_offsets.ptr, d.output_items.ptr, d.status.ptr);
    } else if (d.words_per_row <= bitmap_block_threads * 16) {
        bitmap_row_compact_kernel<16><<<static_cast<unsigned>(d.rows),
            bitmap_block_threads, 0, stream>>>(d.rows, d.edges,
            d.words_per_row, output_capacity, d.union_words.ptr,
            d.word_counts.ptr, d.row_counts.ptr, index.edge_ids.ptr,
            d.output_offsets.ptr, d.output_items.ptr, d.status.ptr);
    } else {
        throw std::runtime_error("two-kernel bitmap path supports at most 2048 words per row");
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_baseline(baseline_device &d, const common_device &c,
    const query_set &q, std::uint64_t rows, std::uint64_t epoch,
    cudaStream_t stream) {
    CUDA_CHECK(ix::cuda_query_reverse_postings_any_v1(d.view, epoch, rows,
        c.query_offsets.ptr, c.query_terms.ptr, q.terms.size(), d.output,
        d.scratch, stream));
}

std::uint64_t check_baseline(const baseline_device &d,
    const std::vector<std::vector<ix::indexed_work_item_v1>> &oracle,
    std::uint64_t rows) {
    CUDA_CHECK(cudaDeviceSynchronize());
    ix::cuda_posting_status_v1 status{};
    std::vector<std::uint64_t> offsets(rows + 1);
    CUDA_CHECK(cudaMemcpy(&status, d.status.ptr, sizeof(status), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(offsets.data(), d.output_offsets.ptr,
        offsets.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    if (status.code != static_cast<std::uint32_t>(ix::cuda_posting_code_v1::success))
        throw std::runtime_error("posting baseline returned status " + std::to_string(status.code));
    std::vector<ix::indexed_work_item_v1> items(status.required);
    if (!items.empty()) CUDA_CHECK(cudaMemcpy(items.data(), d.output_items.ptr,
        items.size() * sizeof(items[0]), cudaMemcpyDeviceToHost));
    if (offsets.back() != status.required) throw std::runtime_error("posting total offset mismatch");
    for (std::uint64_t row = 0; row < rows; ++row) {
        const auto begin = offsets[row], end = offsets[row + 1];
        if (end - begin != oracle[row].size()) throw std::runtime_error("posting oracle row count mismatch");
        for (std::uint64_t i = 0; i < end - begin; ++i)
            if (items[begin + i].logical_edge_id != oracle[row][i].logical_edge_id
                || items[begin + i].logical_edge_ordinal != oracle[row][i].logical_edge_ordinal)
                throw std::runtime_error("posting output differs from scalar oracle");
    }
    return status.required;
}

std::uint64_t check_bitmap(const bitmap_device &d,
    const std::vector<std::vector<ix::indexed_work_item_v1>> &oracle,
    std::uint64_t rows) {
    CUDA_CHECK(cudaDeviceSynchronize());
    candidate_status status{};
    std::vector<std::uint64_t> offsets(rows + 1);
    CUDA_CHECK(cudaMemcpy(&status, d.status.ptr, sizeof(status), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(offsets.data(), d.output_offsets.ptr,
        offsets.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    if (status.code != 0) throw std::runtime_error("bitmap candidate returned status " + std::to_string(status.code));
    std::vector<ix::indexed_work_item_v1> items(status.required);
    if (!items.empty()) CUDA_CHECK(cudaMemcpy(items.data(), d.output_items.ptr,
        items.size() * sizeof(items[0]), cudaMemcpyDeviceToHost));
    if (offsets.back() != status.required) throw std::runtime_error("bitmap total offset mismatch");
    for (std::uint64_t row = 0; row < rows; ++row) {
        const auto begin = offsets[row], end = offsets[row + 1];
        if (end - begin != oracle[row].size()) throw std::runtime_error("bitmap oracle row count mismatch");
        for (std::uint64_t i = 0; i < end - begin; ++i)
            if (items[begin + i].logical_edge_id != oracle[row][i].logical_edge_id
                || items[begin + i].logical_edge_ordinal != oracle[row][i].logical_edge_ordinal)
                throw std::runtime_error("bitmap output differs from scalar oracle");
    }
    return status.required;
}

std::uint64_t check_bitmap_two_kernel(const bitmap_two_kernel_device &d,
    const std::vector<std::vector<ix::indexed_work_item_v1>> &oracle,
    std::uint64_t rows) {
    CUDA_CHECK(cudaDeviceSynchronize());
    candidate_status status{};
    std::vector<std::uint64_t> offsets(rows + 1);
    CUDA_CHECK(cudaMemcpy(&status, d.status.ptr, sizeof(status), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(offsets.data(), d.output_offsets.ptr,
        offsets.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    if (status.code != 0)
        throw std::runtime_error("two-kernel bitmap candidate returned status " + std::to_string(status.code));
    std::vector<ix::indexed_work_item_v1> items(status.required);
    if (!items.empty()) CUDA_CHECK(cudaMemcpy(items.data(), d.output_items.ptr,
        items.size() * sizeof(items[0]), cudaMemcpyDeviceToHost));
    if (offsets.back() != status.required)
        throw std::runtime_error("two-kernel bitmap total offset mismatch");
    for (std::uint64_t row = 0; row < rows; ++row) {
        const auto begin = offsets[row], end = offsets[row + 1];
        if (end - begin != oracle[row].size())
            throw std::runtime_error("two-kernel bitmap oracle row count mismatch");
        for (std::uint64_t i = 0; i < end - begin; ++i)
            if (items[begin + i].logical_edge_id != oracle[row][i].logical_edge_id
                || items[begin + i].logical_edge_ordinal != oracle[row][i].logical_edge_ordinal)
                throw std::runtime_error("two-kernel bitmap output differs from scalar oracle");
    }
    return status.required;
}

struct paired_timing_samples {
    std::vector<double> baseline;
    std::vector<double> bitmap_flat;
    std::vector<double> bitmap_two_kernel;
};

paired_timing_samples time_paired_variants(baseline_device &baseline,
    bitmap_device &flat, bitmap_two_kernel_device &row,
    const common_device &common, const query_set &q,
    const config &cfg, std::uint64_t rows) {
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    for (std::uint32_t i = 0; i < cfg.warmup; ++i) {
        launch_baseline(baseline, common, q, rows, kStructureEpoch, stream);
        launch_bitmap(flat, common, q, flat.output_capacity, stream);
        launch_bitmap_two_kernel(row, flat, common, q, row.output_capacity, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    paired_timing_samples out;
    auto measure = [&](auto &&launch) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (std::uint32_t i = 0; i < cfg.iterations; ++i) launch();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        return static_cast<double>(event_elapsed(start, stop) / cfg.iterations);
    };
    for (std::uint32_t sample = 0; sample < cfg.samples; ++sample) {
        const auto baseline_call = [&] {
            return measure([&] {
                launch_baseline(baseline, common, q, rows, kStructureEpoch, stream);
            });
        };
        const auto flat_call = [&] {
            return measure([&] {
                launch_bitmap(flat, common, q, flat.output_capacity, stream);
            });
        };
        const auto row_call = [&] {
            return measure([&] {
                launch_bitmap_two_kernel(row, flat, common, q,
                    row.output_capacity, stream);
            });
        };
        if (sample % 3 == 0) {
            out.baseline.push_back(baseline_call());
            out.bitmap_flat.push_back(flat_call());
            out.bitmap_two_kernel.push_back(row_call());
        } else if (sample % 3 == 1) {
            out.bitmap_flat.push_back(flat_call());
            out.bitmap_two_kernel.push_back(row_call());
            out.baseline.push_back(baseline_call());
        } else {
            out.bitmap_two_kernel.push_back(row_call());
            out.baseline.push_back(baseline_call());
            out.bitmap_flat.push_back(flat_call());
        }
    }
    cudaEventDestroy(start); cudaEventDestroy(stop); cudaStreamDestroy(stream);
    return out;
}

void test_bitmap_overflow(bitmap_device &d, const common_device &c,
    const query_set &q, std::uint64_t expected) {
    const auto sentinel = ix::indexed_work_item_v1{0xfedcba9876543210ull, 0xfedcba9876543210ull};
    std::vector<ix::indexed_work_item_v1> fill(std::min<std::uint64_t>(expected, 128), sentinel);
    if (!fill.empty()) CUDA_CHECK(cudaMemcpy(d.output_items.ptr, fill.data(),
        fill.size() * sizeof(fill[0]), cudaMemcpyHostToDevice));
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    launch_bitmap(d, c, q, expected == 0 ? 0 : expected - 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    candidate_status status{};
    CUDA_CHECK(cudaMemcpy(&status, d.status.ptr, sizeof(status), cudaMemcpyDeviceToHost));
    if (expected != 0 && (status.code != 1 || status.required != expected))
        throw std::runtime_error("bitmap overflow status did not report exact required count");
    if (expected == 0 && status.code != 0)
        throw std::runtime_error("bitmap empty overflow probe did not succeed");
    if (!fill.empty()) {
        std::vector<ix::indexed_work_item_v1> observed(fill.size());
        CUDA_CHECK(cudaMemcpy(observed.data(), d.output_items.ptr,
            observed.size() * sizeof(observed[0]), cudaMemcpyDeviceToHost));
        for (const auto &x : observed)
            if (x.logical_edge_id != sentinel.logical_edge_id
                || x.logical_edge_ordinal != sentinel.logical_edge_ordinal)
                throw std::runtime_error("bitmap overflow wrote a partial output");
    }
    cudaStreamDestroy(stream);
}

void test_bitmap_two_kernel_overflow(bitmap_two_kernel_device &d,
    const bitmap_device &index, const common_device &c, const query_set &q,
    std::uint64_t expected) {
    const auto sentinel = ix::indexed_work_item_v1{
        0xfedcba9876543210ull, 0xfedcba9876543210ull};
    std::vector<ix::indexed_work_item_v1> fill(
        std::min<std::uint64_t>(expected, 128), sentinel);
    if (!fill.empty()) CUDA_CHECK(cudaMemcpy(d.output_items.ptr, fill.data(),
        fill.size() * sizeof(fill[0]), cudaMemcpyHostToDevice));
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    launch_bitmap_two_kernel(d, index, c, q,
        expected == 0 ? 0 : expected - 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    candidate_status status{};
    CUDA_CHECK(cudaMemcpy(&status, d.status.ptr, sizeof(status), cudaMemcpyDeviceToHost));
    if (expected != 0 && (status.code != 1 || status.required != expected))
        throw std::runtime_error("two-kernel bitmap overflow status mismatch");
    if (expected == 0 && status.code != 0)
        throw std::runtime_error("two-kernel bitmap empty overflow probe failed");
    if (!fill.empty()) {
        std::vector<ix::indexed_work_item_v1> observed(fill.size());
        CUDA_CHECK(cudaMemcpy(observed.data(), d.output_items.ptr,
            observed.size() * sizeof(observed[0]), cudaMemcpyDeviceToHost));
        for (const auto &x : observed)
            if (x.logical_edge_id != sentinel.logical_edge_id
                || x.logical_edge_ordinal != sentinel.logical_edge_ordinal)
                throw std::runtime_error("two-kernel bitmap overflow wrote partial output");
    }
    cudaStreamDestroy(stream);
}

void test_irregular_query_sets(const fixture &f, const posting_projection &p,
    const bitmap_projection &b, const config &cfg) {
    query_set q;
    q.offsets = {0, 2, 2, 4, 5};
    q.terms = {
        {0, 0, f.edges[0].source}, {0, 0, f.edges[0].source},
        {1, 0, f.edges[0].target}, {2, 0, f.edges[1].cofactor},
        {0, 0, 1},
    };
    constexpr std::uint64_t rows = 4;
    const auto oracle = scalar_oracle(f, q);
    auto common = prepare_common(q);
    auto baseline = prepare_baseline(p, common, rows, q.terms.size(), cfg.edges);
    auto bitmap = prepare_bitmap(b, f, common, cfg, rows);
    auto bitmap_two = prepare_bitmap_two_kernel(bitmap, cfg, rows);
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    launch_baseline(baseline, common, q, rows, kStructureEpoch, stream);
    const auto baseline_count = check_baseline(baseline, oracle, rows);
    launch_bitmap(bitmap, common, q, bitmap.output_capacity, stream);
    const auto bitmap_count = check_bitmap(bitmap, oracle, rows);
    if (baseline_count != bitmap_count)
        throw std::runtime_error("irregular query baseline/candidate count mismatch");
    test_bitmap_overflow(bitmap, common, q, bitmap_count);
    launch_bitmap_two_kernel(bitmap_two, bitmap, common, q,
        bitmap_two.output_capacity, stream);
    const auto bitmap_two_count = check_bitmap_two_kernel(bitmap_two, oracle, rows);
    if (baseline_count != bitmap_two_count)
        throw std::runtime_error("irregular query baseline/two-kernel bitmap count mismatch");
    test_bitmap_two_kernel_overflow(bitmap_two, bitmap, common, q, bitmap_two_count);
    CUDA_CHECK(cudaStreamDestroy(stream));

    query_set empty;
    empty.offsets = {0, 0, 0};
    const auto empty_oracle = scalar_oracle(f, empty);
    auto empty_common = prepare_common(empty);
    auto empty_baseline = prepare_baseline(p, empty_common, 2, empty.terms.size(), cfg.edges);
    auto empty_bitmap = prepare_bitmap(b, f, empty_common, cfg, 2);
    auto empty_bitmap_two = prepare_bitmap_two_kernel(empty_bitmap, cfg, 2);
    CUDA_CHECK(cudaStreamCreate(&stream));
    launch_baseline(empty_baseline, empty_common, empty, 2, kStructureEpoch, stream);
    if (check_baseline(empty_baseline, empty_oracle, 2) != 0)
        throw std::runtime_error("empty posting query did not return zero work");
    launch_bitmap(empty_bitmap, empty_common, empty, 0, stream);
    if (check_bitmap(empty_bitmap, empty_oracle, 2) != 0)
        throw std::runtime_error("empty bitmap query did not return zero work");
    launch_bitmap_two_kernel(empty_bitmap_two, empty_bitmap, empty_common,
        empty, 0, stream);
    if (check_bitmap_two_kernel(empty_bitmap_two, empty_oracle, 2) != 0)
        throw std::runtime_error("empty two-kernel bitmap query did not return zero work");
    CUDA_CHECK(cudaStreamDestroy(stream));
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::string hex64(std::uint64_t x) {
    std::ostringstream s; s << "0x" << std::hex << std::setw(16)
        << std::setfill('0') << x; return s.str();
}

std::string samples_json(const std::vector<double> &v) {
    std::ostringstream s; s << '[' << std::setprecision(9);
    for (std::size_t i = 0; i < v.size(); ++i) { if (i) s << ','; s << v[i]; }
    s << ']'; return s.str();
}

void assert_expected_uuid() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    const unsigned char expected[16] = {0x6c,0x1c,0xac,0x7f,0xa3,0x60,0x0a,0xef,
        0xba,0x98,0x28,0x28,0xbf,0xd1,0xdb,0x1a};
    if (props.major != 7 || props.minor != 0
        || std::memcmp(props.uuid.bytes, expected, 16) != 0)
        throw std::runtime_error("benchmark requires the admitted V100 GPU UUID 6c1cac7f-a360-0aef-ba98-2828bfd1db1a");
}

void run_case(const config &cfg) {
    const auto f = make_fixture(cfg);
    const auto q = make_queries(cfg);
    const auto oracle = scalar_oracle(f, q);
    const auto rows = 2 * cfg.cells;
    const auto postings_begin = clock_type::now();
    const auto postings = make_postings(f);
    const double posting_build_ms = std::chrono::duration<double, std::milli>(
        clock_type::now() - postings_begin).count();
    const auto bitmap_begin = clock_type::now();
    const auto bitmaps = make_bitmaps(f, cfg);
    const double bitmap_build_ms = std::chrono::duration<double, std::milli>(
        clock_type::now() - bitmap_begin).count();

    auto common = prepare_common(q);
    auto baseline = prepare_baseline(postings, common, rows, q.terms.size(), cfg.edges);
    auto bitmap = prepare_bitmap(bitmaps, f, common, cfg, rows);
    auto bitmap_two = prepare_bitmap_two_kernel(bitmap, cfg, rows);
    cudaStream_t check_stream{};
    CUDA_CHECK(cudaStreamCreate(&check_stream));
    launch_baseline(baseline, common, q, rows, kStructureEpoch, check_stream);
    const auto baseline_count = check_baseline(baseline, oracle, rows);
    launch_bitmap(bitmap, common, q, bitmap.output_capacity, check_stream);
    const auto bitmap_count = check_bitmap(bitmap, oracle, rows);
    if (baseline_count != bitmap_count) throw std::runtime_error("candidate total differs from baseline");
    test_bitmap_overflow(bitmap, common, q, bitmap_count);
    launch_bitmap_two_kernel(bitmap_two, bitmap, common, q,
        bitmap_two.output_capacity, check_stream);
    const auto bitmap_two_count = check_bitmap_two_kernel(bitmap_two, oracle, rows);
    if (baseline_count != bitmap_two_count)
        throw std::runtime_error("two-kernel bitmap total differs from baseline");
    test_bitmap_two_kernel_overflow(bitmap_two, bitmap, common, q, bitmap_two_count);
    CUDA_CHECK(cudaStreamDestroy(check_stream));
    if (cfg.id == "anchor_e2176_b64")
        test_irregular_query_sets(f, postings, bitmaps, cfg);

    const auto paired_samples = time_paired_variants(baseline, bitmap,
        bitmap_two, common, q, cfg, rows);
    const auto &baseline_samples = paired_samples.baseline;
    const auto &bitmap_samples = paired_samples.bitmap_flat;
    const auto &bitmap_two_samples = paired_samples.bitmap_two_kernel;
    const double baseline_query_ms = median(baseline_samples);
    const double bitmap_query_ms = median(bitmap_samples);
    const double bitmap_two_query_ms = median(bitmap_two_samples);
    const double common_setup_ms = common.allocation_ms + common.h2d_ms;
    const double baseline_setup_ms = posting_build_ms + common_setup_ms
        + baseline.allocation_ms + baseline.h2d_ms;
    const double bitmap_setup_ms = bitmap_build_ms + common_setup_ms
        + bitmap.allocation_ms + bitmap.h2d_ms;
    const double bitmap_two_setup_ms = bitmap_build_ms + common_setup_ms
        + bitmap.index_allocation_ms + bitmap.h2d_ms
        + bitmap_two.allocation_ms;
    const auto baseline_memory = common.memory_bytes + baseline.index_bytes
        + baseline.workspace_output_bytes;
    const auto bitmap_memory = common.memory_bytes + bitmap.index_bytes
        + bitmap.workspace_output_bytes;
    const auto bitmap_two_memory = common.memory_bytes + bitmap_two.index_bytes
        + bitmap_two.workspace_output_bytes;

    std::cout << std::setprecision(9)
        << "{\"schema\":\"cellerator.sparse_support.bitmap_retrieval.v2\","
        << "\"case_id\":\"" << cfg.id << "\",\"fixture_digest\":\""
        << hex64(combined_fixture_digest(cfg, f, q))
        << "\",\"support_digest\":\"" << hex64(f.digest)
        << "\",\"seed\":" << kSeed
        << ",\"edges\":" << cfg.edges << ",\"cells\":" << cfg.cells
        << ",\"entities\":" << cfg.entities << ",\"source_skew\":" << cfg.skew
        << ",\"query_rows\":" << rows << ",\"posting_terms\":" << postings.terms.size()
        << ",\"posting_records\":" << postings.items.size()
        << ",\"candidate_count\":" << bitmap_count
        << ",\"correctness\":\"scalar_oracle_exact_and_overflow_pass\""
        << ",\"warmup\":" << cfg.warmup << ",\"samples\":" << cfg.samples
        << ",\"iterations_per_sample\":" << cfg.iterations
        << ",\"timing_policy\":\"paired CUDA-event rounds; candidate order rotates across samples\""
        << ",\"baseline_p50_query_ms\":" << baseline_query_ms
        << ",\"bitmap_p50_query_ms\":" << bitmap_query_ms
        << ",\"bitmap_two_kernel_p50_query_ms\":" << bitmap_two_query_ms
        << ",\"baseline_samples_ms\":" << samples_json(baseline_samples)
        << ",\"bitmap_samples_ms\":" << samples_json(bitmap_samples)
        << ",\"bitmap_two_kernel_samples_ms\":" << samples_json(bitmap_two_samples)
        << ",\"setup_ms\":{\"baseline_posting_cpu_build\":" << posting_build_ms
        << ",\"bitmap_cpu_build\":" << bitmap_build_ms
        << ",\"common_allocation_and_h2d\":" << common_setup_ms
        << ",\"baseline_allocation\":" << baseline.allocation_ms
        << ",\"baseline_h2d\":" << baseline.h2d_ms
        << ",\"bitmap_allocation\":" << bitmap.allocation_ms
        << ",\"bitmap_index_allocation\":" << bitmap.index_allocation_ms
        << ",\"bitmap_workspace_allocation\":" << bitmap.workspace_allocation_ms
        << ",\"bitmap_h2d\":" << bitmap.h2d_ms
        << ",\"bitmap_two_kernel_workspace_allocation\":" << bitmap_two.allocation_ms
        << ",\"baseline_cold_total\":" << baseline_setup_ms
        << ",\"bitmap_cold_total\":" << bitmap_setup_ms
        << ",\"bitmap_two_kernel_cold_total\":" << bitmap_two_setup_ms << '}'
        << ",\"memory_bytes\":{\"common_query_and_ids\":" << common.memory_bytes
        << ",\"baseline_index\":" << baseline.index_bytes
        << ",\"bitmap_index\":" << bitmap.index_bytes
        << ",\"baseline_workspace_output\":" << baseline.workspace_output_bytes
        << ",\"bitmap_workspace_output\":" << bitmap.workspace_output_bytes
        << ",\"bitmap_two_kernel_workspace_output\":" << bitmap_two.workspace_output_bytes
        << ",\"baseline_total\":" << baseline_memory
        << ",\"bitmap_total\":" << bitmap_memory
        << ",\"bitmap_two_kernel_total\":" << bitmap_two_memory << '}'
        << ",\"amortized_ms\":{\"reuse_1_baseline\":" << baseline_setup_ms + baseline_query_ms
        << ",\"reuse_1_bitmap\":" << bitmap_setup_ms + bitmap_query_ms
        << ",\"reuse_1_bitmap_two_kernel\":" << bitmap_two_setup_ms + bitmap_two_query_ms
        << ",\"reuse_16_baseline\":" << baseline_setup_ms / 16 + baseline_query_ms
        << ",\"reuse_16_bitmap\":" << bitmap_setup_ms / 16 + bitmap_query_ms
        << ",\"reuse_16_bitmap_two_kernel\":" << bitmap_two_setup_ms / 16 + bitmap_two_query_ms << '}'
        << ",\"device\":{\"name\":\"Tesla V100-SXM2-16GB\",\"uuid\":\"6c1cac7f-a360-0aef-ba98-2828bfd1db1a\",\"sm\":\"70\",\"runtime\":\"CUDA 12.9\"}"
        << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        std::vector<config> matrix = {
            {"anchor_e2176_b64", 2176, 64, 256, 0.0},
            {"scale_e65536_b1", 65536, 1, 256, 0.0},
            {"source_skew_0_9", 2176, 64, 256, 0.9},
            {"batch_b1", 2176, 1, 256, 0.0},
        };
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--dry-run") {
                for (const auto &cfg : matrix) {
                    const auto fixture = make_fixture(cfg);
                    const auto queries = make_queries(cfg);
                    const auto oracle = scalar_oracle(fixture, queries);
                    std::uint64_t count = 0;
                    for (const auto &row : oracle) count += row.size();
                    std::cout << "{\"case_id\":\"" << cfg.id
                        << "\",\"fixture_digest\":\""
                        << hex64(combined_fixture_digest(cfg, fixture, queries))
                        << "\",\"support_digest\":\"" << hex64(fixture.digest)
                        << "\",\"scalar_candidate_count\":" << count << "}\n";
                }
                return 0;
            } else if (arg == "--case" && i + 1 < argc) {
                const std::string wanted = argv[++i];
                matrix.erase(std::remove_if(matrix.begin(), matrix.end(), [&](const config &c) {
                    return c.id != wanted;
                }), matrix.end());
                if (matrix.empty()) throw std::runtime_error("unknown case: " + wanted);
            } else if (arg == "--samples" && i + 1 < argc) {
                const auto value = static_cast<std::uint32_t>(std::stoul(argv[++i]));
                if (value < 3 || value > 1000) throw std::runtime_error("samples outside 3..1000");
                for (auto &c : matrix) c.samples = value;
            } else if (arg == "--iterations" && i + 1 < argc) {
                const auto value = static_cast<std::uint32_t>(std::stoul(argv[++i]));
                if (value == 0 || value > 1000) throw std::runtime_error("iterations outside 1..1000");
                for (auto &c : matrix) c.iterations = value;
            } else if (arg == "--help") {
                std::cout << "bitmap_retrieval_bench [--dry-run] [--case CASE] [--samples N] [--iterations N]\n";
                return 0;
            } else throw std::runtime_error("unknown argument: " + arg);
        }
        if (matrix.empty()) throw std::runtime_error("empty matrix");
        assert_expected_uuid();
        for (const auto &cfg : matrix) run_case(cfg);
        CUDA_CHECK(cudaDeviceSynchronize());
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "bitmap retrieval experiment failed: " << e.what() << '\n';
        return 1;
    }
}
