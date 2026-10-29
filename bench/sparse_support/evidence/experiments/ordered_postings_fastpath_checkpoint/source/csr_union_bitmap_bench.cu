#include <Cellerator/compute/indexing/cuda_postings.cuh>
#include <Cellerator/compute/indexing/reverse_postings.hh>
#include <Cellerator/compute/support/logical_support.hh>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ix = cellerator::compute::indexing;
namespace support = cellerator::compute::support;
using clock_type = std::chrono::steady_clock;

#define CUDA_CHECK(expr) do { \
    const cudaError_t error_ = (expr); \
    if (error_ != cudaSuccess) \
        throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(error_)); \
} while (false)

namespace {

constexpr std::uint64_t kEdges = 1'000'000;
constexpr std::uint64_t kEntities = 20'000;
constexpr std::uint64_t kCells = 32;
constexpr unsigned kThreads = 256;
constexpr unsigned kTileEdges = kThreads;
constexpr unsigned kWarpsPerScatterBlock = 4;

struct device_status {
    std::uint32_t code = 0;
    std::uint32_t reserved = 0;
    std::uint64_t required = 0;
};

struct input_case {
    std::string name;
    std::uint64_t q_per_cell = 0;
    std::vector<std::uint64_t> query_offsets;
    std::vector<std::uint64_t> query_entities;
    std::vector<std::uint64_t> oracle_offsets;
    std::vector<std::uint64_t> oracle_ordinals;
    std::vector<std::uint64_t> raw_posting_counts;
    std::uint64_t output_capacity = 0;
};

template<class T>
class device_buffer {
public:
    device_buffer() = default;
    explicit device_buffer(std::uint64_t count) { reset(count); }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    device_buffer(device_buffer &&other) noexcept { swap(other); }
    device_buffer &operator=(device_buffer &&other) noexcept {
        if (this != &other) { release(); swap(other); }
        return *this;
    }
    ~device_buffer() { release(); }

    void reset(std::uint64_t count) {
        release();
        count_ = count;
        if (count_) CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr_),
                                         count_ * sizeof(T)));
    }
    T *get() const noexcept { return ptr_; }
    std::uint64_t size() const noexcept { return count_; }
    std::uint64_t bytes() const noexcept { return count_ * sizeof(T); }

private:
    void release() noexcept {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        count_ = 0;
    }
    void swap(device_buffer &other) noexcept {
        std::swap(ptr_, other.ptr_);
        std::swap(count_, other.count_);
    }
    T *ptr_ = nullptr;
    std::uint64_t count_ = 0;
};

template<class T>
std::vector<T> read_raw(const fs::path &path, std::uint64_t count) {
    const auto bytes = count * sizeof(T);
    if (!fs::is_regular_file(path) || fs::file_size(path) != bytes)
        throw std::runtime_error("raw input size mismatch: " + path.string());
    std::vector<T> values(count);
    std::ifstream input(path, std::ios::binary);
    if (!input || (bytes && !input.read(reinterpret_cast<char *>(values.data()),
                                        static_cast<std::streamsize>(bytes))))
        throw std::runtime_error("cannot read raw input: " + path.string());
    return values;
}

std::vector<std::uint64_t> read_u64(const fs::path &dir, const char *name,
                                    std::uint64_t count) {
    return read_raw<std::uint64_t>(dir / (std::string(name) + ".bin"), count);
}

template<class T>
void copy_to_device(device_buffer<T> &dst, const std::vector<T> &src,
                    cudaStream_t stream) {
    if (src.size() > dst.size()) throw std::runtime_error("device upload exceeds allocation");
    if (!src.empty()) CUDA_CHECK(cudaMemcpyAsync(dst.get(), src.data(),
        src.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
}

void check_launch() { CUDA_CHECK(cudaGetLastError()); }

__global__ void scatter_csr_to_edge_bits(
    const std::uint64_t *query_offsets, const std::uint64_t *query_entities,
    const std::uint64_t *source_indptr, const std::uint64_t *source_ordinals,
    const std::uint32_t *rank_by_ordinal,
    bool identity_id_order, std::uint64_t cells, std::uint64_t q_per_cell,
    std::uint64_t words_per_cell,
    std::uint32_t *edge_bits) {
    const std::uint64_t group_count = (q_per_cell + kWarpsPerScatterBlock - 1)
        / kWarpsPerScatterBlock;
    const std::uint64_t cell = static_cast<std::uint64_t>(blockIdx.x) / group_count;
    const std::uint64_t group = static_cast<std::uint64_t>(blockIdx.x) % group_count;
    if (cell >= cells) return;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const std::uint64_t term_in_cell = group * kWarpsPerScatterBlock + warp;
    if (term_in_cell >= q_per_cell) return;
    const std::uint64_t query_index = query_offsets[cell] + term_in_cell;
    if (query_index >= query_offsets[cell + 1]) return;
    const std::uint64_t entity = query_entities[query_index];
    const std::uint64_t begin = source_indptr[entity];
    const std::uint64_t end = source_indptr[entity + 1];
    for (std::uint64_t item = begin + lane; item < end; item += 32u) {
        const std::uint64_t ordinal = source_ordinals[item];
        const std::uint32_t rank = identity_id_order
            ? static_cast<std::uint32_t>(ordinal) : rank_by_ordinal[ordinal];
        const std::uint64_t word = cell * words_per_cell + (rank >> 5);
        atomicOr(edge_bits + word, 1u << (rank & 31u));
    }
}

__global__ void edge_bit_popcounts(const std::uint32_t *edge_bits,
                                   std::uint64_t *word_counts,
                                   std::uint64_t word_count) {
    const std::uint64_t word = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (word < word_count) word_counts[word] = __popc(edge_bits[word]);
}

__global__ void finish_prefix_and_offsets(
    std::uint64_t cells, std::uint64_t words_per_cell,
    const std::uint64_t *word_counts, const std::uint64_t *word_prefix,
    std::uint64_t total_words, std::uint64_t output_capacity,
    std::uint64_t *output_offsets, device_status *status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const std::uint64_t required = total_words == 0 ? 0
        : word_prefix[total_words - 1] + word_counts[total_words - 1];
    status->required = required;
    status->reserved = 0;
    if (required > output_capacity) {
        status->code = 1u;
        return;
    }
    status->code = 0u;
    output_offsets[0] = 0;
    for (std::uint64_t cell = 0; cell < cells; ++cell) {
        const std::uint64_t last = (cell + 1) * words_per_cell - 1;
        output_offsets[cell + 1] = word_prefix[last] + word_counts[last];
    }
}

__global__ void count_concat_terms(
    const std::uint64_t *query_entities, std::uint64_t query_count,
    const std::uint64_t *source_indptr, std::uint64_t *term_counts) {
    const std::uint64_t term = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (term >= query_count) return;
    const std::uint64_t entity = query_entities[term];
    term_counts[term] = source_indptr[entity + 1] - source_indptr[entity];
}

__global__ void finish_concat_offsets(
    std::uint64_t cells, std::uint64_t q_per_cell,
    const std::uint64_t *term_counts, const std::uint64_t *term_prefix,
    std::uint64_t query_count, std::uint64_t output_capacity,
    std::uint64_t *output_offsets, device_status *status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const std::uint64_t required = query_count == 0 ? 0
        : term_prefix[query_count - 1] + term_counts[query_count - 1];
    status->required = required;
    status->reserved = 0;
    if (required > output_capacity) {
        status->code = 1u;
        return;
    }
    status->code = 0u;
    output_offsets[0] = 0;
    if (q_per_cell == 0) {
        for (std::uint64_t cell = 0; cell < cells; ++cell)
            output_offsets[cell + 1] = 0;
        return;
    }
    for (std::uint64_t cell = 0; cell < cells; ++cell) {
        const std::uint64_t last = (cell + 1) * q_per_cell - 1;
        output_offsets[cell + 1] = term_prefix[last] + term_counts[last];
    }
}

__global__ void scatter_concat_terms(
    const std::uint64_t *query_entities, std::uint64_t query_count,
    const std::uint64_t *source_indptr, const std::uint64_t *source_ordinals,
    const std::uint64_t *term_prefix, const std::uint64_t *edge_ids,
    const device_status *status, ix::indexed_work_item_v1 *output) {
    if (status->code != 0u) return;
    const std::uint64_t term = blockIdx.x;
    if (term >= query_count) return;
    const std::uint64_t entity = query_entities[term];
    const std::uint64_t begin = source_indptr[entity];
    const std::uint64_t end = source_indptr[entity + 1];
    for (std::uint64_t p = begin + threadIdx.x; p < end; p += blockDim.x) {
        const std::uint64_t ordinal = source_ordinals[p];
        output[term_prefix[term] + p - begin] = {edge_ids[ordinal], ordinal};
    }
}

__global__ void emit_edge_bits(
    const std::uint32_t *edge_bits, const std::uint64_t *word_prefix,
    const std::uint64_t *edge_ids,
    const std::uint64_t *edge_ids_by_rank,
    const std::uint64_t *edge_ordinals_by_rank, bool identity_id_order,
    std::uint64_t edges,
    std::uint64_t words_per_cell, std::uint64_t total_words,
    const device_status *status, ix::indexed_work_item_v1 *output) {
    const std::uint64_t word_index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (word_index >= total_words || status->code != 0u) return;
    std::uint32_t bits = edge_bits[word_index];
    std::uint64_t output_index = word_prefix[word_index];
    const std::uint64_t ordinal_base = (word_index % words_per_cell) * 32;
    while (bits != 0u) {
        const unsigned bit = __ffs(static_cast<int>(bits)) - 1u;
        const std::uint64_t rank = ordinal_base + bit;
        if (rank < edges) output[output_index++] = {
            identity_id_order ? edge_ids[rank] : edge_ids_by_rank[rank],
            identity_id_order ? rank : edge_ordinals_by_rank[rank]};
        bits &= bits - 1u;
    }
}

__global__ void fill_entity_membership(
    const std::uint64_t *query_entities, std::uint64_t query_count,
    std::uint64_t q_per_cell, std::uint64_t words_per_cell,
    std::uint32_t *membership_bits) {
    const std::uint64_t query = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (query >= query_count || q_per_cell == 0) return;
    const std::uint64_t cell = query / q_per_cell;
    const std::uint64_t entity = query_entities[query];
    atomicOr(membership_bits + cell * words_per_cell + (entity >> 5),
             1u << (entity & 31u));
}

using full_reduce = cub::BlockReduce<std::uint32_t, kThreads>;
__global__ void count_full_scan_tiles(
    const std::uint32_t *source_by_ordinal,
    const std::uint32_t *membership_bits,
    std::uint64_t cells, std::uint64_t edges,
    std::uint64_t membership_words_per_cell,
    std::uint64_t tiles_per_cell, std::uint64_t *tile_counts) {
    __shared__ typename full_reduce::TempStorage reduce_storage;
    const std::uint64_t block = blockIdx.x;
    const std::uint64_t cell = block / tiles_per_cell;
    const std::uint64_t tile = block % tiles_per_cell;
    if (cell >= cells) return;
    const std::uint64_t ordinal = tile * kTileEdges + threadIdx.x;
    std::uint32_t hit = 0;
    if (ordinal < edges) {
        const std::uint64_t entity = source_by_ordinal[ordinal];
        hit = (membership_bits[cell * membership_words_per_cell + (entity >> 5)]
            >> (entity & 31u)) & 1u;
    }
    const std::uint32_t total = full_reduce(reduce_storage).Sum(hit);
    if (threadIdx.x == 0) tile_counts[block] = total;
}

using full_scan = cub::BlockScan<std::uint32_t, kThreads>;
__global__ void emit_full_scan_tiles(
    const std::uint32_t *source_by_ordinal,
    const std::uint32_t *membership_bits,
    const std::uint64_t *block_prefix,
    const std::uint64_t *edge_ids_by_rank,
    const std::uint64_t *edge_ordinals_by_rank,
    bool identity_id_order,
    std::uint64_t cells, std::uint64_t edges,
    std::uint64_t membership_words_per_cell,
    std::uint64_t tiles_per_cell,
    const device_status *status,
    ix::indexed_work_item_v1 *output) {
    if (status->code != 0u) return;
    __shared__ typename full_scan::TempStorage scan_storage;
    const std::uint64_t block = blockIdx.x;
    const std::uint64_t cell = block / tiles_per_cell;
    const std::uint64_t tile = block % tiles_per_cell;
    if (cell >= cells) return;
    const std::uint64_t ordinal = tile * kTileEdges + threadIdx.x;
    std::uint32_t hit = 0;
    if (ordinal < edges) {
        const std::uint64_t entity = source_by_ordinal[ordinal];
        hit = (membership_bits[cell * membership_words_per_cell + (entity >> 5)]
            >> (entity & 31u)) & 1u;
    }
    std::uint32_t local_offset = 0;
    full_scan(scan_storage).ExclusiveSum(hit, local_offset);
    if (hit) output[block_prefix[block] + local_offset] = {
        edge_ids_by_rank[ordinal],
        identity_id_order ? ordinal : edge_ordinals_by_rank[ordinal]};
}

void launch_bitmap(
    const std::uint64_t *query_offsets, const std::uint64_t *query_entities,
    const std::uint64_t *source_indptr, const std::uint64_t *source_ordinals,
    const std::uint32_t *rank_by_ordinal,
    bool identity_id_order,
    std::uint64_t cells, std::uint64_t q_per_cell, std::uint64_t edges,
    std::uint64_t words_per_cell, std::uint64_t total_words,
    std::uint64_t output_capacity, std::uint32_t *edge_bits,
    std::uint64_t *word_counts, std::uint64_t *word_prefix,
    void *scan_temp, std::size_t scan_temp_bytes,
    std::uint64_t *output_offsets, device_status *status,
    const std::uint64_t *edge_ids,
    const std::uint64_t *edge_ids_by_rank,
    const std::uint64_t *edge_ordinals_by_rank,
    ix::indexed_work_item_v1 *output,
    cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(edge_bits, 0,
        total_words * sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(status, 0, sizeof(device_status), stream));
    const std::uint64_t groups = (q_per_cell + kWarpsPerScatterBlock - 1)
        / kWarpsPerScatterBlock;
    const std::uint64_t grid = cells * groups;
    if (grid) {
        scatter_csr_to_edge_bits<<<static_cast<unsigned>(grid),
            kWarpsPerScatterBlock * 32, 0, stream>>>(query_offsets,
            query_entities, source_indptr, source_ordinals, rank_by_ordinal,
            identity_id_order, cells,
            q_per_cell, words_per_cell, edge_bits);
        check_launch();
    }
    edge_bit_popcounts<<<static_cast<unsigned>((total_words + kThreads - 1) / kThreads),
        kThreads, 0, stream>>>(edge_bits, word_counts, total_words);
    check_launch();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(scan_temp, scan_temp_bytes,
        word_counts, word_prefix, total_words, stream));
    finish_prefix_and_offsets<<<1, 1, 0, stream>>>(cells, words_per_cell,
        word_counts, word_prefix, total_words, output_capacity,
        output_offsets, status);
    check_launch();
    emit_edge_bits<<<static_cast<unsigned>((total_words + kThreads - 1) / kThreads),
        kThreads, 0, stream>>>(edge_bits, word_prefix, edge_ids,
        edge_ids_by_rank, edge_ordinals_by_rank, identity_id_order, edges,
        words_per_cell, total_words, status, output);
    check_launch();
}

void launch_full_scan(
    const std::uint64_t *query_entities, std::uint64_t q_per_cell,
    const std::uint32_t *source_by_rank,
    const std::uint64_t *edge_ids_by_rank,
    const std::uint64_t *edge_ordinals_by_rank,
    bool identity_id_order,
    std::uint64_t cells, std::uint64_t edges,
    std::uint64_t membership_words_per_cell, std::uint64_t total_membership_words,
    std::uint64_t tiles_per_cell, std::uint64_t tile_count,
    std::uint64_t output_capacity, std::uint32_t *membership_bits,
    std::uint64_t *tile_counts, std::uint64_t *tile_prefix,
    void *scan_temp, std::size_t scan_temp_bytes,
    std::uint64_t *output_offsets, device_status *status,
    ix::indexed_work_item_v1 *output, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(membership_bits, 0,
        total_membership_words * sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(status, 0, sizeof(device_status), stream));
    const std::uint64_t query_count = cells * q_per_cell;
    if (query_count) {
        fill_entity_membership<<<static_cast<unsigned>((query_count + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(query_entities, query_count, q_per_cell,
            membership_words_per_cell, membership_bits);
        check_launch();
    }
    count_full_scan_tiles<<<static_cast<unsigned>(tile_count), kThreads, 0, stream>>>(
        source_by_rank, membership_bits, cells, edges,
        membership_words_per_cell, tiles_per_cell, tile_counts);
    check_launch();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(scan_temp, scan_temp_bytes,
        tile_counts, tile_prefix, tile_count, stream));
    finish_prefix_and_offsets<<<1, 1, 0, stream>>>(cells, tiles_per_cell,
        tile_counts, tile_prefix, tile_count, output_capacity,
        output_offsets, status);
    check_launch();
    emit_full_scan_tiles<<<static_cast<unsigned>(tile_count), kThreads, 0, stream>>>(
        source_by_rank, membership_bits, tile_prefix, edge_ids_by_rank,
        edge_ordinals_by_rank, identity_id_order, cells,
        edges, membership_words_per_cell, tiles_per_cell, status, output);
    check_launch();
}

struct support_inputs {
    std::vector<std::uint64_t> edge_ids;
    std::vector<std::uint64_t> edge_ids_by_rank;
    std::vector<std::uint64_t> edge_ordinals_by_rank;
    std::vector<std::uint32_t> rank_by_ordinal;
    std::vector<std::uint64_t> source_entities;
    std::vector<std::uint64_t> participant_entities;
    std::vector<std::uint64_t> fixture_source_indptr;
    std::vector<std::uint64_t> fixture_source_ordinals;
};

bool concat_applicable(const input_case &in, const support_inputs &support);

support_inputs load_support(const fs::path &root) {
    support_inputs in;
    const fs::path dir = root / "support";
    in.edge_ids = read_u64(dir, "logical_edge_id", kEdges);
    in.source_entities = read_u64(dir, "source_entity_id", kEdges);
    in.participant_entities = read_u64(dir, "participant_entity_id", kEdges);
    in.fixture_source_indptr = read_u64(dir, "source_edge_indptr", kEntities + 1);
    in.fixture_source_ordinals = read_u64(dir, "source_edge_ordinals", kEdges);
    in.edge_ordinals_by_rank.resize(kEdges);
    std::iota(in.edge_ordinals_by_rank.begin(), in.edge_ordinals_by_rank.end(), 0u);
    std::sort(in.edge_ordinals_by_rank.begin(), in.edge_ordinals_by_rank.end(),
        [&](std::uint64_t a, std::uint64_t b) {
            return in.edge_ids[a] < in.edge_ids[b];
        });
    in.edge_ids_by_rank.resize(kEdges);
    in.rank_by_ordinal.resize(kEdges);
    for (std::uint64_t ordinal = 0; ordinal < kEdges; ++ordinal) {
        if (in.source_entities[ordinal] >= kEntities
            || in.participant_entities[ordinal] >= kEntities
            || in.source_entities[ordinal] == in.participant_entities[ordinal])
            throw std::runtime_error("fixture logical edge/participant identity check failed");
    }
    for (std::uint64_t rank = 0; rank < kEdges; ++rank) {
        const auto ordinal = in.edge_ordinals_by_rank[rank];
        if (rank != 0 && in.edge_ids_by_rank[rank - 1] >= in.edge_ids[ordinal])
            throw std::runtime_error("logical edge IDs are not unique");
        in.edge_ids_by_rank[rank] = in.edge_ids[ordinal];
        in.rank_by_ordinal[ordinal] = static_cast<std::uint32_t>(rank);
    }
    return in;
}

input_case load_case(const fs::path &root, const std::string &name) {
    input_case in;
    in.name = name;
    const fs::path dir = root / "queries" / name;
    in.query_offsets = read_u64(dir, "query_offsets", kCells + 1);
    in.query_entities = read_u64(dir, "query_terms", in.query_offsets.back());
    in.oracle_offsets = read_u64(dir, "oracle_offsets", kCells + 1);
    in.oracle_ordinals = read_u64(dir, "oracle_edge_ordinals", in.oracle_offsets.back());
    in.raw_posting_counts = read_u64(dir, "raw_posting_count_per_row", kCells);
    in.q_per_cell = in.query_offsets[1] - in.query_offsets[0];
    if (in.query_offsets[0] != 0 || in.oracle_offsets[0] != 0
        || in.query_offsets.back() != kCells * in.q_per_cell)
        throw std::runtime_error("query CSR offsets are not canonical");
    for (std::uint64_t row = 0; row < kCells; ++row) {
        if (in.query_offsets[row + 1] - in.query_offsets[row] != in.q_per_cell)
            throw std::runtime_error("query rows have unexpected lengths");
        if (in.oracle_offsets[row + 1] < in.oracle_offsets[row])
            throw std::runtime_error("oracle offsets are not monotonic");
    }
    in.output_capacity = std::accumulate(in.raw_posting_counts.begin(),
        in.raw_posting_counts.end(), std::uint64_t{0});
    if (in.oracle_offsets.back() > in.output_capacity)
        throw std::runtime_error("raw posting upper bound is below exact union");
    for (std::uint64_t row = 0; row < kCells; ++row)
        for (std::uint64_t i = in.oracle_offsets[row] + 1;
             i < in.oracle_offsets[row + 1]; ++i)
            if (in.oracle_ordinals[i] < in.oracle_ordinals[i - 1])
                throw std::runtime_error("CPU oracle ordinals are not stable sorted");
    return in;
}

struct direct_csr {
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> ordinals;
};

direct_csr build_direct_csr(const std::vector<std::uint64_t> &source_entities) {
    direct_csr csr;
    csr.offsets.assign(kEntities + 1, 0);
    for (const auto entity : source_entities) ++csr.offsets[entity + 1];
    for (std::uint64_t entity = 0; entity < kEntities; ++entity)
        csr.offsets[entity + 1] += csr.offsets[entity];
    csr.ordinals.resize(source_entities.size());
    std::vector<std::uint64_t> cursor(csr.offsets.begin(), csr.offsets.end() - 1);
    for (std::uint64_t ordinal = 0; ordinal < source_entities.size(); ++ordinal)
        csr.ordinals[cursor[source_entities[ordinal]]++] = ordinal;
    return csr;
}

void build_baseline_postings(
    const support_inputs &input,
    std::vector<support::logical_support_edge_v1> &edges,
    std::vector<support::logical_support_participant_v1> &participants,
    ix::reverse_posting_view_v1 *out,
    std::vector<ix::reverse_posting_record_v1> &entry_storage) {
    edges.resize(kEdges);
    participants.resize(kEdges);
    for (std::uint64_t ordinal = 0; ordinal < kEdges; ++ordinal) {
        edges[ordinal] = {input.edge_ids[ordinal], ordinal, 1u, 0u};
        participants[ordinal] = {0u, 0u, input.source_entities[ordinal]};
    }
    const support::logical_support_view_v1 logical{
        support::logical_support_schema_version_v1, 0u, 1u, 1u, 1u,
        edges.data(), edges.size(), participants.data(), participants.size()};
    if (support::validate_logical_support_v1(logical)
        != support::logical_support_status_v1::success)
        throw std::runtime_error("logical support fixture failed validation");
    const support::support_snapshot_pair_v1 snapshot{&logical, nullptr, 1u};
    const auto req = ix::query_reverse_posting_requirements_v1(snapshot);
    std::vector<ix::reverse_posting_record_v1> scratch(req.record_capacity);
    entry_storage.resize(req.maximum_unique_entries);
    const auto result = ix::build_reverse_postings_v1(snapshot,
        scratch.data(), scratch.size(), entry_storage.data(), entry_storage.size(), out);
    if (result != ix::posting_status_v1::success)
        throw std::runtime_error("host reverse-posting index build failed");
}

struct baseline_index_host {
    std::vector<ix::posting_term_v1> terms;
    std::vector<std::uint64_t> offsets;
    std::vector<ix::indexed_work_item_v1> items;
};

baseline_index_host build_cuda_projection(const ix::reverse_posting_view_v1 &host) {
    const auto req = ix::cuda_posting_projection_requirements_v1(host);
    baseline_index_host projection;
    projection.terms.resize(req.term_count);
    projection.offsets.resize(req.term_count + 1);
    projection.items.resize(req.item_count);
    ix::cuda_posting_prepare_requirements_v1 observed{};
    const auto status = ix::build_cuda_posting_projection_v1(host,
        projection.terms.data(), projection.terms.size(),
        projection.offsets.data(), projection.offsets.size(),
        projection.items.data(), projection.items.size(), &observed);
    if (status != ix::cuda_posting_prepare_status_v1::success
        || observed.term_count != req.term_count || observed.item_count != req.item_count)
        throw std::runtime_error("host CUDA projection build failed");
    return projection;
}

void assert_direct_csr_matches_fixture(
    const direct_csr &csr, const support_inputs &input) {
    if (csr.offsets != input.fixture_source_indptr
        || csr.ordinals != input.fixture_source_ordinals)
        throw std::runtime_error("direct CSR build differs from scaled fixture CSR");
}

std::uint64_t max_raw_capacity(const fs::path &root) {
    const std::vector<std::string> names = {
        "q1_unique", "q32_unique", "q32_duplicate4", "q128_unique",
        "q128_duplicate4", "q1024_unique", "q1024_duplicate4",
        "q2048_unique", "q2048_duplicate4"};
    std::uint64_t max_capacity = 0;
    for (const auto &name : names) {
        const auto dir = root / "queries" / name;
        const auto raw = read_u64(dir, "raw_posting_count_per_row", kCells);
        max_capacity = std::max(max_capacity, std::accumulate(raw.begin(), raw.end(),
            std::uint64_t{0}));
    }
    return max_capacity;
}

void validate_cpu_fixture(const fs::path &root, const direct_csr &csr,
                          const support_inputs &support) {
    const std::vector<std::string> names = {
        "q1_unique", "q32_unique", "q32_duplicate4", "q128_unique",
        "q128_duplicate4", "q1024_unique", "q1024_duplicate4",
        "q2048_unique", "q2048_duplicate4"};
    std::vector<std::uint8_t> selected(kEdges);
    for (const auto &name : names) {
        const auto in = load_case(root, name);
        const bool can_concat = concat_applicable(in, support);
        const bool has_duplicate_terms = name.find("duplicate") != std::string::npos;
        if (can_concat == has_duplicate_terms)
            throw std::runtime_error(name + " count-scan-concat applicability mismatch");
        for (std::uint64_t cell = 0; cell < kCells; ++cell) {
            std::fill(selected.begin(), selected.end(), 0);
            for (std::uint64_t qi = in.query_offsets[cell];
                 qi < in.query_offsets[cell + 1]; ++qi) {
                const auto entity = in.query_entities[qi];
                for (std::uint64_t p = csr.offsets[entity];
                     p < csr.offsets[entity + 1]; ++p)
                    selected[csr.ordinals[p]] = 1;
            }
            std::vector<std::uint64_t> result_ordinals;
            for (std::uint64_t ordinal = 0; ordinal < kEdges; ++ordinal) {
                if (selected[ordinal]) result_ordinals.push_back(ordinal);
            }
            std::sort(result_ordinals.begin(), result_ordinals.end(),
                [&](std::uint64_t a, std::uint64_t b) {
                    return support.edge_ids[a] < support.edge_ids[b];
                });
            const auto oracle_begin = in.oracle_ordinals.begin() + in.oracle_offsets[cell];
            const auto oracle_end = in.oracle_ordinals.begin() + in.oracle_offsets[cell + 1];
            if (!std::equal(result_ordinals.begin(), result_ordinals.end(),
                    oracle_begin, oracle_end))
                throw std::runtime_error(name + " CPU CSR union differs from frozen oracle");

            if (can_concat) {
                std::vector<std::uint64_t> concatenated;
                for (std::uint64_t qi = in.query_offsets[cell];
                     qi < in.query_offsets[cell + 1]; ++qi) {
                    const auto entity = in.query_entities[qi];
                    concatenated.insert(concatenated.end(),
                        csr.ordinals.begin() + csr.offsets[entity],
                        csr.ordinals.begin() + csr.offsets[entity + 1]);
                }
                const auto begin = in.oracle_ordinals.begin() + in.oracle_offsets[cell];
                const auto end = in.oracle_ordinals.begin() + in.oracle_offsets[cell + 1];
                if (!std::equal(concatenated.begin(), concatenated.end(), begin, end))
                    throw std::runtime_error(name + " count-scan-concat CPU oracle mismatch");
            }
        }
    }

    // Independently scramble stable IDs and verify the ID-rank bitmap projection.
    std::vector<std::uint64_t> independent_ids(kEdges);
    std::iota(independent_ids.begin(), independent_ids.end(), std::uint64_t{1});
    std::mt19937_64 rng(0x243f6a8885a308d3ull);
    std::shuffle(independent_ids.begin(), independent_ids.end(), rng);
    support_inputs independent_support = support;
    independent_support.edge_ids = independent_ids;
    for (const auto &name : names) {
        const auto in = load_case(root, name);
        if (concat_applicable(in, independent_support))
            throw std::runtime_error(name + " concat precondition ignored independent ID order");
    }
    std::vector<std::uint64_t> id_order(kEdges);
    std::iota(id_order.begin(), id_order.end(), std::uint64_t{0});
    std::sort(id_order.begin(), id_order.end(), [&](std::uint64_t a, std::uint64_t b) {
        return independent_ids[a] < independent_ids[b];
    });
    std::vector<std::uint32_t> rank_by_ordinal(kEdges);
    for (std::uint64_t rank = 0; rank < kEdges; ++rank)
        rank_by_ordinal[id_order[rank]] = static_cast<std::uint32_t>(rank);
    for (const auto &name : names) {
        const auto in = load_case(root, name);
        for (std::uint64_t cell = 0; cell < kCells; ++cell) {
            std::fill(selected.begin(), selected.end(), 0);
            for (std::uint64_t qi = in.query_offsets[cell];
                 qi < in.query_offsets[cell + 1]; ++qi) {
                const auto entity = in.query_entities[qi];
                for (std::uint64_t p = csr.offsets[entity];
                     p < csr.offsets[entity + 1]; ++p)
                    selected[csr.ordinals[p]] = 1;
            }
            std::vector<std::uint8_t> rank_bits(kEdges);
            for (std::uint64_t ordinal = 0; ordinal < kEdges; ++ordinal)
                if (selected[ordinal]) rank_bits[rank_by_ordinal[ordinal]] = 1;
            std::vector<std::uint64_t> result_ordinals;
            for (std::uint64_t rank = 0; rank < kEdges; ++rank)
                if (rank_bits[rank]) result_ordinals.push_back(id_order[rank]);
            std::vector<std::uint64_t> expected;
            for (std::uint64_t ordinal = 0; ordinal < kEdges; ++ordinal)
                if (selected[ordinal]) expected.push_back(ordinal);
            std::sort(expected.begin(), expected.end(), [&](std::uint64_t a, std::uint64_t b) {
                return independent_ids[a] < independent_ids[b];
            });
            if (result_ordinals != expected)
                throw std::runtime_error(name + " independent-ID rank bitmap oracle mismatch");
        }
    }
}

void validate_output(
    const char *algorithm, const input_case &in,
    const std::vector<std::uint64_t> &edge_ids,
    const std::uint64_t *offsets_device,
    const ix::indexed_work_item_v1 *items_device,
    const device_status *status_device) {
    device_status status{};
    std::vector<std::uint64_t> offsets(kCells + 1);
    CUDA_CHECK(cudaMemcpy(&status, status_device, sizeof(status), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(offsets.data(), offsets_device,
        offsets.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    if (status.code != 0u || status.required != in.oracle_ordinals.size())
        throw std::runtime_error(std::string(algorithm) + " status/count mismatch");
    if (offsets != in.oracle_offsets)
        throw std::runtime_error(std::string(algorithm) + " row offsets differ from CPU oracle");
    std::vector<ix::indexed_work_item_v1> items(status.required);
    if (status.required) CUDA_CHECK(cudaMemcpy(items.data(), items_device,
        items.size() * sizeof(items[0]), cudaMemcpyDeviceToHost));
    for (std::uint64_t i = 0; i < status.required; ++i) {
        const auto ordinal = in.oracle_ordinals[i];
        if (items[i].logical_edge_ordinal != ordinal
            || items[i].logical_edge_id != edge_ids[ordinal])
            throw std::runtime_error(std::string(algorithm) + " output differs from CPU oracle");
    }
}

double event_time_ms(cudaEvent_t start, cudaEvent_t stop) {
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    return elapsed;
}

template<class Launch>
double timed_query(Launch &&launch, cudaEvent_t start, cudaEvent_t stop,
                   cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    launch();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    return event_time_ms(start, stop);
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

double percentile(const std::vector<double> &values, double fraction) {
    if (values.empty()) return 0.0;
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const auto index = static_cast<std::size_t>(fraction * (sorted.size() - 1));
    return sorted[index];
}

std::uint64_t serial_merge_work_proxy(const input_case &in) {
    std::uint64_t eligible = 0;
    for (std::uint64_t cell = 0; cell < kCells; ++cell)
        eligible += in.oracle_offsets[cell + 1] - in.oracle_offsets[cell];
    return in.q_per_cell * eligible;
}

struct benchmark_context {
    std::uint64_t edges = kEdges;
    std::uint64_t entities = kEntities;
    std::uint64_t cells = kCells;
    std::uint64_t words_per_cell = (kEdges + 31) / 32;
    std::uint64_t total_edge_words = 0;
    std::uint64_t membership_words_per_cell = (kEntities + 31) / 32;
    std::uint64_t total_membership_words = 0;
    std::uint64_t tiles_per_cell = (kEdges + kTileEdges - 1) / kTileEdges;
    std::uint64_t tile_count = 0;
    std::uint64_t output_capacity = 0;
    std::uint64_t query_capacity = 0;
    bool identity_id_order = false;

    device_buffer<std::uint64_t> query_offsets;
    device_buffer<std::uint64_t> query_entities;
    device_buffer<ix::posting_term_v1> posting_queries;
    device_buffer<ix::posting_term_v1> index_terms;
    device_buffer<std::uint64_t> index_term_offsets;
    device_buffer<ix::indexed_work_item_v1> index_items;
    device_buffer<std::uint64_t> source_indptr;
    device_buffer<std::uint64_t> source_ordinals;
    device_buffer<std::uint64_t> edge_ids;
    device_buffer<std::uint64_t> edge_ids_by_rank;
    device_buffer<std::uint64_t> edge_ordinals_by_rank;
    device_buffer<std::uint32_t> rank_by_ordinal;
    device_buffer<std::uint32_t> source_by_rank;

    device_buffer<std::uint64_t> output_offsets;
    device_buffer<ix::indexed_work_item_v1> output_items;
    device_buffer<ix::cuda_posting_scratch_v1::cursor_v1> posting_cursors;
    device_buffer<std::uint64_t> posting_counts;
    device_buffer<ix::cuda_posting_status_v1> posting_status;

    device_buffer<std::uint32_t> edge_bits;
    device_buffer<std::uint64_t> edge_word_counts;
    device_buffer<std::uint64_t> edge_word_prefix;
    device_buffer<device_status> edge_status;
    device_buffer<std::uint64_t> concat_term_counts;
    device_buffer<std::uint64_t> concat_term_prefix;
    device_buffer<device_status> concat_status;
    device_buffer<std::uint32_t> entity_membership;
    device_buffer<std::uint64_t> tile_counts;
    device_buffer<std::uint64_t> tile_prefix;
    device_buffer<device_status> full_status;
    device_buffer<std::uint8_t> scan_temp;
    std::size_t edge_scan_temp_bytes = 0;
    std::size_t tile_scan_temp_bytes = 0;
    std::size_t concat_scan_temp_bytes = 0;

    std::uint64_t baseline_index_bytes = 0;
    std::uint64_t candidate_index_bytes = 0;
    std::uint64_t concat_index_bytes = 0;
    std::uint64_t fullscan_index_bytes = 0;
    std::uint64_t baseline_scratch_bytes = 0;
    std::uint64_t candidate_scratch_bytes = 0;
    std::uint64_t concat_scratch_bytes = 0;
    std::uint64_t fullscan_scratch_bytes = 0;
};

void initialize_context(
    benchmark_context &ctx,
    const support_inputs &input,
    const direct_csr &direct,
    const baseline_index_host &baseline,
    std::uint64_t output_capacity,
    cudaStream_t stream) {
    ctx.total_edge_words = ctx.cells * ctx.words_per_cell;
    ctx.total_membership_words = ctx.cells * ctx.membership_words_per_cell;
    ctx.tile_count = ctx.cells * ctx.tiles_per_cell;
    ctx.output_capacity = output_capacity;
    ctx.identity_id_order = true;
    for (std::uint64_t ordinal = 0; ordinal < input.rank_by_ordinal.size(); ++ordinal)
        ctx.identity_id_order &= input.rank_by_ordinal[ordinal] == ordinal;

    ctx.query_offsets.reset(ctx.cells + 1);
    ctx.query_entities.reset(ctx.query_capacity);
    ctx.posting_queries.reset(ctx.query_capacity);
    ctx.index_terms.reset(baseline.terms.size());
    ctx.index_term_offsets.reset(baseline.offsets.size());
    ctx.index_items.reset(baseline.items.size());
    ctx.source_indptr.reset(direct.offsets.size());
    ctx.source_ordinals.reset(direct.ordinals.size());
    ctx.edge_ids.reset(input.edge_ids.size());
    ctx.edge_ids_by_rank.reset(ctx.identity_id_order ? 0 : input.edge_ids_by_rank.size());
    ctx.edge_ordinals_by_rank.reset(ctx.identity_id_order ? 0 : input.edge_ordinals_by_rank.size());
    ctx.rank_by_ordinal.reset(ctx.identity_id_order ? 0 : input.rank_by_ordinal.size());
    ctx.source_by_rank.reset(input.source_entities.size());

    ctx.output_offsets.reset(ctx.cells + 1);
    ctx.output_items.reset(output_capacity);
    ctx.posting_cursors.reset(ctx.query_capacity);
    ctx.posting_counts.reset(ctx.cells);
    ctx.posting_status.reset(1);

    ctx.edge_bits.reset(ctx.total_edge_words);
    ctx.edge_word_counts.reset(ctx.total_edge_words);
    ctx.edge_word_prefix.reset(ctx.total_edge_words);
    ctx.edge_status.reset(1);
    ctx.concat_term_counts.reset(ctx.query_capacity);
    ctx.concat_term_prefix.reset(ctx.query_capacity);
    ctx.concat_status.reset(1);
    ctx.entity_membership.reset(ctx.total_membership_words);
    ctx.tile_counts.reset(ctx.tile_count);
    ctx.tile_prefix.reset(ctx.tile_count);
    ctx.full_status.reset(1);

    std::size_t edge_temp = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, edge_temp,
        ctx.edge_word_counts.get(), ctx.edge_word_prefix.get(),
        ctx.total_edge_words, stream));
    std::size_t tile_temp = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, tile_temp,
        ctx.tile_counts.get(), ctx.tile_prefix.get(), ctx.tile_count, stream));
    std::size_t concat_temp = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, concat_temp,
        ctx.concat_term_counts.get(), ctx.concat_term_prefix.get(),
        ctx.query_capacity, stream));
    ctx.edge_scan_temp_bytes = edge_temp;
    ctx.tile_scan_temp_bytes = tile_temp;
    ctx.concat_scan_temp_bytes = concat_temp;
    const auto common_temp = std::max({edge_temp, tile_temp, concat_temp});
    ctx.scan_temp.reset(common_temp);

    std::vector<std::uint32_t> source_by_rank(input.source_entities.size());
    for (std::uint64_t rank = 0; rank < input.edge_ordinals_by_rank.size(); ++rank)
        source_by_rank[rank] = static_cast<std::uint32_t>(
            input.source_entities[input.edge_ordinals_by_rank[rank]]);
    copy_to_device(ctx.index_terms, baseline.terms, stream);
    copy_to_device(ctx.index_term_offsets, baseline.offsets, stream);
    copy_to_device(ctx.index_items, baseline.items, stream);
    copy_to_device(ctx.source_indptr, direct.offsets, stream);
    copy_to_device(ctx.source_ordinals, direct.ordinals, stream);
    copy_to_device(ctx.edge_ids, input.edge_ids, stream);
    if (!ctx.identity_id_order) {
        copy_to_device(ctx.edge_ids_by_rank, input.edge_ids_by_rank, stream);
        copy_to_device(ctx.edge_ordinals_by_rank, input.edge_ordinals_by_rank, stream);
        copy_to_device(ctx.rank_by_ordinal, input.rank_by_ordinal, stream);
    }
    copy_to_device(ctx.source_by_rank, source_by_rank, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ctx.baseline_index_bytes = ctx.index_terms.bytes() + ctx.index_term_offsets.bytes()
        + ctx.index_items.bytes();
    ctx.candidate_index_bytes = ctx.source_indptr.bytes() + ctx.source_ordinals.bytes()
        + (ctx.identity_id_order ? ctx.edge_ids.bytes()
            : ctx.edge_ids_by_rank.bytes() + ctx.edge_ordinals_by_rank.bytes()
                + ctx.rank_by_ordinal.bytes());
    ctx.concat_index_bytes = ctx.source_indptr.bytes() + ctx.source_ordinals.bytes()
        + ctx.edge_ids.bytes();
    ctx.fullscan_index_bytes = ctx.source_by_rank.bytes()
        + (ctx.identity_id_order ? ctx.edge_ids.bytes()
            : ctx.edge_ids_by_rank.bytes() + ctx.edge_ordinals_by_rank.bytes());
    ctx.baseline_scratch_bytes = ctx.posting_cursors.bytes() + ctx.posting_counts.bytes()
        + ctx.posting_status.bytes();
    ctx.candidate_scratch_bytes = ctx.edge_bits.bytes() + ctx.edge_word_counts.bytes()
        + ctx.edge_word_prefix.bytes() + ctx.edge_status.bytes()
        + ctx.edge_scan_temp_bytes;
    ctx.concat_scratch_bytes = ctx.concat_term_counts.bytes()
        + ctx.concat_term_prefix.bytes() + ctx.concat_status.bytes()
        + ctx.concat_scan_temp_bytes;
    ctx.fullscan_scratch_bytes = ctx.entity_membership.bytes() + ctx.tile_counts.bytes()
        + ctx.tile_prefix.bytes() + ctx.full_status.bytes() + ctx.tile_scan_temp_bytes;
}

void upload_query_case(benchmark_context &ctx, const input_case &in,
                       cudaStream_t stream, double *upload_ms) {
    if (in.query_entities.size() > ctx.query_capacity)
        throw std::runtime_error("query case exceeds allocated query capacity");
    std::vector<ix::posting_term_v1> postings(in.query_entities.size());
    for (std::size_t i = 0; i < in.query_entities.size(); ++i)
        postings[i] = {0u, 0u, in.query_entities[i]};
    const auto begin = clock_type::now();
    copy_to_device(ctx.query_offsets, in.query_offsets, stream);
    copy_to_device(ctx.query_entities, in.query_entities, stream);
    copy_to_device(ctx.posting_queries, postings, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    *upload_ms = std::chrono::duration<double, std::milli>(
        clock_type::now() - begin).count();
}

void run_postings(benchmark_context &ctx, const input_case &in,
                  std::uint64_t capacity, cudaStream_t stream) {
    const ix::cuda_reverse_posting_view_v1 view{
        1u, 1u, 1u, ctx.index_terms.get(), ctx.index_term_offsets.get(),
        ctx.index_terms.size(), ctx.index_items.get(), ctx.index_items.size()};
    const ix::cuda_posting_output_v1 output{
        ctx.output_offsets.get(), ctx.cells + 1,
        ctx.output_items.get(), capacity};
    const ix::cuda_posting_scratch_v1 scratch{
        ctx.posting_cursors.get(), ctx.posting_cursors.size(),
        ctx.posting_counts.get(), ctx.posting_status.get()};
    CUDA_CHECK(ix::cuda_query_reverse_postings_any_v1(view, 1u, ctx.cells,
        ctx.query_offsets.get(), ctx.posting_queries.get(), in.query_entities.size(),
        output, scratch, stream));
}

void run_concat(benchmark_context &ctx, const input_case &in,
                std::uint64_t capacity, cudaStream_t stream) {
    const std::uint64_t query_count = in.query_entities.size();
    CUDA_CHECK(cudaMemsetAsync(ctx.concat_status.get(), 0,
        sizeof(device_status), stream));
    count_concat_terms<<<static_cast<unsigned>((query_count + kThreads - 1) / kThreads),
        kThreads, 0, stream>>>(ctx.query_entities.get(), query_count,
        ctx.source_indptr.get(), ctx.concat_term_counts.get());
    check_launch();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(ctx.scan_temp.get(),
        ctx.concat_scan_temp_bytes, ctx.concat_term_counts.get(),
        ctx.concat_term_prefix.get(), query_count, stream));
    finish_concat_offsets<<<1, 1, 0, stream>>>(ctx.cells, in.q_per_cell,
        ctx.concat_term_counts.get(), ctx.concat_term_prefix.get(), query_count,
        capacity, ctx.output_offsets.get(), ctx.concat_status.get());
    check_launch();
    scatter_concat_terms<<<static_cast<unsigned>(query_count), kThreads, 0, stream>>>(
        ctx.query_entities.get(), query_count, ctx.source_indptr.get(),
        ctx.source_ordinals.get(), ctx.concat_term_prefix.get(),
        ctx.edge_ids.get(), ctx.concat_status.get(), ctx.output_items.get());
    check_launch();
}

bool concat_applicable(const input_case &in, const support_inputs &support) {
    if (in.q_per_cell == 0 || support.edge_ids.size() != kEdges
        || support.source_entities.size() != kEdges)
        return false;
    for (std::uint64_t ordinal = 1; ordinal < kEdges; ++ordinal) {
        if (support.edge_ids[ordinal] <= support.edge_ids[ordinal - 1]
            || support.source_entities[ordinal] < support.source_entities[ordinal - 1])
            return false;
    }
    for (std::uint64_t cell = 0; cell < kCells; ++cell) {
        const auto begin = in.query_offsets[cell];
        const auto end = in.query_offsets[cell + 1];
        for (std::uint64_t i = begin + 1; i < end; ++i)
            if (in.query_entities[i] <= in.query_entities[i - 1]) return false;
    }
    return true;
}

void run_bitmap(benchmark_context &ctx, const input_case &in,
                std::uint64_t capacity, cudaStream_t stream) {
    launch_bitmap(ctx.query_offsets.get(), ctx.query_entities.get(),
        ctx.source_indptr.get(), ctx.source_ordinals.get(),
        ctx.rank_by_ordinal.get(), ctx.identity_id_order, ctx.cells,
        in.q_per_cell, ctx.edges, ctx.words_per_cell, ctx.total_edge_words,
        capacity, ctx.edge_bits.get(), ctx.edge_word_counts.get(),
        ctx.edge_word_prefix.get(), ctx.scan_temp.get(), ctx.edge_scan_temp_bytes,
        ctx.output_offsets.get(), ctx.edge_status.get(),
        ctx.edge_ids.get(), ctx.edge_ids_by_rank.get(),
        ctx.edge_ordinals_by_rank.get(),
        ctx.output_items.get(), stream);
}

void run_full(benchmark_context &ctx, const input_case &in,
              std::uint64_t capacity, cudaStream_t stream) {
    launch_full_scan(ctx.query_entities.get(), in.q_per_cell,
        ctx.source_by_rank.get(),
        ctx.identity_id_order ? ctx.edge_ids.get() : ctx.edge_ids_by_rank.get(),
        ctx.edge_ordinals_by_rank.get(), ctx.identity_id_order, ctx.cells, ctx.edges,
        ctx.membership_words_per_cell, ctx.total_membership_words,
        ctx.tiles_per_cell, ctx.tile_count, capacity, ctx.entity_membership.get(),
        ctx.tile_counts.get(), ctx.tile_prefix.get(), ctx.scan_temp.get(),
        ctx.tile_scan_temp_bytes, ctx.output_offsets.get(), ctx.full_status.get(),
        ctx.output_items.get(), stream);
}

void verify_no_partial_output(benchmark_context &ctx, cudaStream_t stream,
                              const char *algorithm) {
    std::vector<std::uint8_t> observed(ctx.output_items.bytes());
    if (!observed.empty()) CUDA_CHECK(cudaMemcpy(observed.data(), ctx.output_items.get(),
        observed.size(), cudaMemcpyDeviceToHost));
    if (std::any_of(observed.begin(), observed.end(), [](std::uint8_t x) { return x != 0xA5; }))
        throw std::runtime_error(std::string(algorithm) + " overflow wrote partial output items");
    std::vector<std::uint8_t> observed_offsets((ctx.cells + 1) * sizeof(std::uint64_t));
    CUDA_CHECK(cudaMemcpy(observed_offsets.data(), ctx.output_offsets.get(),
        observed_offsets.size(), cudaMemcpyDeviceToHost));
    if (std::any_of(observed_offsets.begin(), observed_offsets.end(),
        [](std::uint8_t x) { return x != 0xA5; }))
        throw std::runtime_error(std::string(algorithm) + " overflow wrote partial row offsets");
    (void)stream;
}

void test_all_overflows(benchmark_context &ctx, const input_case &in,
                        bool run_postings_overflow, bool run_concat_overflow,
                        cudaStream_t stream) {
    const std::uint64_t required = in.oracle_ordinals.size();
    if (required == 0 || required > ctx.output_items.size())
        throw std::runtime_error("overflow probe requires a nonempty exact result");
    const std::uint64_t capacity = required - 1;
    if (run_postings_overflow) {
        CUDA_CHECK(cudaMemsetAsync(ctx.output_items.get(), 0xA5,
            ctx.output_items.bytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(ctx.output_offsets.get(), 0xA5,
            (ctx.cells + 1) * sizeof(std::uint64_t), stream));
        run_postings(ctx, in, capacity, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ix::cuda_posting_status_v1 posting_status{};
        CUDA_CHECK(cudaMemcpy(&posting_status, ctx.posting_status.get(),
            sizeof(posting_status), cudaMemcpyDeviceToHost));
        if (posting_status.code != static_cast<std::uint32_t>(ix::cuda_posting_code_v1::output_overflow)
            || posting_status.required != required)
            throw std::runtime_error("posting overflow did not report exact required count");
        verify_no_partial_output(ctx, stream, "postings");
    }

    if (run_concat_overflow) {
        CUDA_CHECK(cudaMemsetAsync(ctx.output_items.get(), 0xA5,
            ctx.output_items.bytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(ctx.output_offsets.get(), 0xA5,
            (ctx.cells + 1) * sizeof(std::uint64_t), stream));
        run_concat(ctx, in, capacity, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        device_status concat_status{};
        CUDA_CHECK(cudaMemcpy(&concat_status, ctx.concat_status.get(),
            sizeof(concat_status), cudaMemcpyDeviceToHost));
        if (concat_status.code != 1u || concat_status.required != required)
            throw std::runtime_error("concat overflow did not report exact required count");
        verify_no_partial_output(ctx, stream, "concat");
    }

    CUDA_CHECK(cudaMemsetAsync(ctx.output_items.get(), 0xA5,
        ctx.output_items.bytes(), stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.output_offsets.get(), 0xA5,
        (ctx.cells + 1) * sizeof(std::uint64_t), stream));
    run_bitmap(ctx, in, capacity, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    device_status candidate_status{};
    CUDA_CHECK(cudaMemcpy(&candidate_status, ctx.edge_status.get(),
        sizeof(candidate_status), cudaMemcpyDeviceToHost));
    if (candidate_status.code != 1u || candidate_status.required != required)
        throw std::runtime_error("bitmap overflow did not report exact required count");
    verify_no_partial_output(ctx, stream, "CSR bitmap");

    CUDA_CHECK(cudaMemsetAsync(ctx.output_items.get(), 0xA5,
        ctx.output_items.bytes(), stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.output_offsets.get(), 0xA5,
        (ctx.cells + 1) * sizeof(std::uint64_t), stream));
    run_full(ctx, in, capacity, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    device_status full_status{};
    CUDA_CHECK(cudaMemcpy(&full_status, ctx.full_status.get(),
        sizeof(full_status), cudaMemcpyDeviceToHost));
    if (full_status.code != 1u || full_status.required != required)
        throw std::runtime_error("full-scan overflow did not report exact required count");
    verify_no_partial_output(ctx, stream, "parallel full scan");
}

void assert_expected_uuid() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    const unsigned char expected[16] = {
        0x6c, 0x1c, 0xac, 0x7f, 0xa3, 0x60, 0x0a, 0xef,
        0xba, 0x98, 0x28, 0x28, 0xbf, 0xd1, 0xdb, 0x1a};
    if (props.major != 7 || props.minor != 0
        || std::memcmp(props.uuid.bytes, expected, 16) != 0)
        throw std::runtime_error("requires admitted V100 GPU UUID 6c1cac7f-a360-0aef-ba98-2828bfd1db1a");
}

void test_empty_query(benchmark_context &ctx, const input_case &in,
                      cudaStream_t stream) {
    input_case empty;
    empty.name = "empty_query";
    empty.q_per_cell = 0;
    empty.query_offsets.assign(kCells + 1, 0);
    empty.oracle_offsets.assign(kCells + 1, 0);
    empty.raw_posting_counts.assign(kCells, 0);
    empty.output_capacity = 0;
    empty.query_entities.clear();
    empty.oracle_ordinals.clear();
    double upload_ms = 0.0;
    upload_query_case(ctx, empty, stream, &upload_ms);
    run_postings(ctx, empty, 0, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    device_status poster{};
    ix::cuda_posting_status_v1 pstatus{};
    CUDA_CHECK(cudaMemcpy(&pstatus, ctx.posting_status.get(), sizeof(pstatus), cudaMemcpyDeviceToHost));
    if (pstatus.code != static_cast<std::uint32_t>(ix::cuda_posting_code_v1::success)
        || pstatus.required != 0u)
        throw std::runtime_error("empty posting query failed");
    run_bitmap(ctx, empty, 0, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&poster, ctx.edge_status.get(), sizeof(poster), cudaMemcpyDeviceToHost));
    if (poster.code != 0u || poster.required != 0u)
        throw std::runtime_error("empty bitmap query failed");
    run_full(ctx, empty, 0, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&poster, ctx.full_status.get(), sizeof(poster), cudaMemcpyDeviceToHost));
    if (poster.code != 0u || poster.required != 0u)
        throw std::runtime_error("empty full-scan query failed");
    upload_query_case(ctx, in, stream, &upload_ms);
}

void timed_stages_bitmap(benchmark_context &ctx, const input_case &in,
                         std::vector<std::vector<double>> &samples,
                         cudaStream_t stream) {
    constexpr int stage_count = 6;
    std::vector<cudaEvent_t> starts(stage_count), stops(stage_count);
    for (int i = 0; i < stage_count; ++i) {
        CUDA_CHECK(cudaEventCreate(&starts[i]));
        CUDA_CHECK(cudaEventCreate(&stops[i]));
    }
    const std::uint64_t groups = (in.q_per_cell + kWarpsPerScatterBlock - 1)
        / kWarpsPerScatterBlock;
    const std::uint64_t grid = ctx.cells * groups;
    for (int sample = 0; sample < 5; ++sample) {
        CUDA_CHECK(cudaEventRecord(starts[0], stream));
        CUDA_CHECK(cudaMemsetAsync(ctx.edge_bits.get(), 0,
            ctx.total_edge_words * sizeof(std::uint32_t), stream));
        CUDA_CHECK(cudaMemsetAsync(ctx.edge_status.get(), 0, sizeof(device_status), stream));
        CUDA_CHECK(cudaEventRecord(stops[0], stream));

        CUDA_CHECK(cudaEventRecord(starts[1], stream));
        scatter_csr_to_edge_bits<<<static_cast<unsigned>(grid),
            kWarpsPerScatterBlock * 32, 0, stream>>>(ctx.query_offsets.get(),
            ctx.query_entities.get(), ctx.source_indptr.get(), ctx.source_ordinals.get(),
            ctx.rank_by_ordinal.get(), ctx.identity_id_order, ctx.cells, in.q_per_cell,
            ctx.words_per_cell, ctx.edge_bits.get());
        check_launch();
        CUDA_CHECK(cudaEventRecord(stops[1], stream));

        CUDA_CHECK(cudaEventRecord(starts[2], stream));
        edge_bit_popcounts<<<static_cast<unsigned>((ctx.total_edge_words + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(ctx.edge_bits.get(), ctx.edge_word_counts.get(),
            ctx.total_edge_words);
        check_launch();
        CUDA_CHECK(cudaEventRecord(stops[2], stream));

        CUDA_CHECK(cudaEventRecord(starts[3], stream));
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(ctx.scan_temp.get(),
            ctx.edge_scan_temp_bytes, ctx.edge_word_counts.get(),
            ctx.edge_word_prefix.get(), ctx.total_edge_words, stream));
        CUDA_CHECK(cudaEventRecord(stops[3], stream));

        CUDA_CHECK(cudaEventRecord(starts[4], stream));
        finish_prefix_and_offsets<<<1, 1, 0, stream>>>(ctx.cells,
            ctx.words_per_cell, ctx.edge_word_counts.get(), ctx.edge_word_prefix.get(),
            ctx.total_edge_words, in.output_capacity, ctx.output_offsets.get(),
            ctx.edge_status.get());
        check_launch();
        CUDA_CHECK(cudaEventRecord(stops[4], stream));

        CUDA_CHECK(cudaEventRecord(starts[5], stream));
        emit_edge_bits<<<static_cast<unsigned>((ctx.total_edge_words + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(ctx.edge_bits.get(), ctx.edge_word_prefix.get(),
            ctx.edge_ids.get(), ctx.edge_ids_by_rank.get(),
            ctx.edge_ordinals_by_rank.get(), ctx.identity_id_order,
            ctx.edges, ctx.words_per_cell,
            ctx.total_edge_words, ctx.edge_status.get(), ctx.output_items.get());
        check_launch();
        CUDA_CHECK(cudaEventRecord(stops[5], stream));
        CUDA_CHECK(cudaEventSynchronize(stops[5]));
        for (int stage = 0; stage < stage_count; ++stage)
            samples[stage].push_back(event_time_ms(starts[stage], stops[stage]) * 1000.0);
    }
    for (int i = 0; i < stage_count; ++i) {
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(stops[i]);
    }
}

void timed_stages_full(benchmark_context &ctx, const input_case &in,
                       std::vector<std::vector<double>> &samples,
                       cudaStream_t stream) {
    constexpr int stage_count = 5;
    std::vector<cudaEvent_t> starts(stage_count), stops(stage_count);
    for (int i = 0; i < stage_count; ++i) {
        CUDA_CHECK(cudaEventCreate(&starts[i]));
        CUDA_CHECK(cudaEventCreate(&stops[i]));
    }
    const std::uint64_t query_count = ctx.cells * in.q_per_cell;
    for (int sample = 0; sample < 5; ++sample) {
        CUDA_CHECK(cudaEventRecord(starts[0], stream));
        CUDA_CHECK(cudaMemsetAsync(ctx.entity_membership.get(), 0,
            ctx.total_membership_words * sizeof(std::uint32_t), stream));
        CUDA_CHECK(cudaMemsetAsync(ctx.full_status.get(), 0, sizeof(device_status), stream));
        CUDA_CHECK(cudaEventRecord(stops[0], stream));

        CUDA_CHECK(cudaEventRecord(starts[1], stream));
        fill_entity_membership<<<static_cast<unsigned>((query_count + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(ctx.query_entities.get(), query_count,
            in.q_per_cell, ctx.membership_words_per_cell, ctx.entity_membership.get());
        check_launch();
        CUDA_CHECK(cudaEventRecord(stops[1], stream));

        CUDA_CHECK(cudaEventRecord(starts[2], stream));
        count_full_scan_tiles<<<static_cast<unsigned>(ctx.tile_count), kThreads, 0, stream>>>(
            ctx.source_by_rank.get(), ctx.entity_membership.get(), ctx.cells,
            ctx.edges, ctx.membership_words_per_cell, ctx.tiles_per_cell,
            ctx.tile_counts.get());
        check_launch();
        CUDA_CHECK(cudaEventRecord(stops[2], stream));

        CUDA_CHECK(cudaEventRecord(starts[3], stream));
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(ctx.scan_temp.get(),
            ctx.tile_scan_temp_bytes, ctx.tile_counts.get(), ctx.tile_prefix.get(),
            ctx.tile_count, stream));
        finish_prefix_and_offsets<<<1, 1, 0, stream>>>(ctx.cells,
            ctx.tiles_per_cell, ctx.tile_counts.get(), ctx.tile_prefix.get(),
            ctx.tile_count, in.output_capacity, ctx.output_offsets.get(),
            ctx.full_status.get());
        check_launch();
        CUDA_CHECK(cudaEventRecord(stops[3], stream));

        CUDA_CHECK(cudaEventRecord(starts[4], stream));
        emit_full_scan_tiles<<<static_cast<unsigned>(ctx.tile_count), kThreads, 0, stream>>>(
            ctx.source_by_rank.get(), ctx.entity_membership.get(),
            ctx.tile_prefix.get(),
            ctx.identity_id_order ? ctx.edge_ids.get() : ctx.edge_ids_by_rank.get(),
            ctx.edge_ordinals_by_rank.get(), ctx.identity_id_order, ctx.cells, ctx.edges,
            ctx.membership_words_per_cell, ctx.tiles_per_cell,
            ctx.full_status.get(), ctx.output_items.get());
        check_launch();
        CUDA_CHECK(cudaEventRecord(stops[4], stream));
        CUDA_CHECK(cudaEventSynchronize(stops[4]));
        for (int stage = 0; stage < stage_count; ++stage)
            samples[stage].push_back(event_time_ms(starts[stage], stops[stage]) * 1000.0);
    }
    for (int i = 0; i < stage_count; ++i) {
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(stops[i]);
    }
}

void timed_stages_concat(benchmark_context &ctx, const input_case &in,
                         std::vector<std::vector<double>> &samples,
                         cudaStream_t stream) {
    constexpr int stage_count = 3;
    std::vector<cudaEvent_t> starts(stage_count), stops(stage_count);
    for (int i = 0; i < stage_count; ++i) {
        CUDA_CHECK(cudaEventCreate(&starts[i]));
        CUDA_CHECK(cudaEventCreate(&stops[i]));
    }
    const std::uint64_t query_count = in.query_entities.size();
    for (int sample = 0; sample < 5; ++sample) {
        CUDA_CHECK(cudaEventRecord(starts[0], stream));
        CUDA_CHECK(cudaMemsetAsync(ctx.concat_status.get(), 0,
            sizeof(device_status), stream));
        count_concat_terms<<<static_cast<unsigned>((query_count + kThreads - 1) / kThreads),
            kThreads, 0, stream>>>(ctx.query_entities.get(), query_count,
            ctx.source_indptr.get(), ctx.concat_term_counts.get());
        check_launch();
        CUDA_CHECK(cudaEventRecord(stops[0], stream));

        CUDA_CHECK(cudaEventRecord(starts[1], stream));
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(ctx.scan_temp.get(),
            ctx.concat_scan_temp_bytes, ctx.concat_term_counts.get(),
            ctx.concat_term_prefix.get(), query_count, stream));
        finish_concat_offsets<<<1, 1, 0, stream>>>(ctx.cells, in.q_per_cell,
            ctx.concat_term_counts.get(), ctx.concat_term_prefix.get(), query_count,
            in.output_capacity, ctx.output_offsets.get(), ctx.concat_status.get());
        check_launch();
        CUDA_CHECK(cudaEventRecord(stops[1], stream));

        CUDA_CHECK(cudaEventRecord(starts[2], stream));
        scatter_concat_terms<<<static_cast<unsigned>(query_count), kThreads, 0, stream>>>(
            ctx.query_entities.get(), query_count, ctx.source_indptr.get(),
            ctx.source_ordinals.get(), ctx.concat_term_prefix.get(),
            ctx.edge_ids.get(), ctx.concat_status.get(), ctx.output_items.get());
        check_launch();
        CUDA_CHECK(cudaEventRecord(stops[2], stream));
        CUDA_CHECK(cudaEventSynchronize(stops[2]));
        for (int stage = 0; stage < stage_count; ++stage)
            samples[stage].push_back(event_time_ms(starts[stage], stops[stage]) * 1000.0);
    }
    for (int i = 0; i < stage_count; ++i) {
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(stops[i]);
    }
}

void print_samples(const std::vector<double> &values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << std::setprecision(8) << values[i];
    }
    std::cout << ']';
}

void run_case(benchmark_context &ctx, const fs::path &root,
              const std::string &name, int warmups, int sample_count,
              const support_inputs &support,
              const std::vector<std::uint64_t> &edge_ids,
              cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop,
              bool overflow_probe) {
    input_case in = load_case(root, name);
    double query_upload_ms = 0.0;
    upload_query_case(ctx, in, stream, &query_upload_ms);

    const bool measure_postings = true;
    if (measure_postings) {
        run_postings(ctx, in, in.output_capacity, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        validate_output("postings", in, edge_ids, ctx.output_offsets.get(),
            ctx.output_items.get(), reinterpret_cast<const device_status *>(ctx.posting_status.get()));
    }

    const bool use_concat = concat_applicable(in, support);
    if (use_concat) {
        run_concat(ctx, in, in.output_capacity, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        validate_output("count-scan-concat", in, edge_ids, ctx.output_offsets.get(),
            ctx.output_items.get(), ctx.concat_status.get());
    }

    run_bitmap(ctx, in, in.output_capacity, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    validate_output("CSR bitmap", in, edge_ids, ctx.output_offsets.get(),
        ctx.output_items.get(), ctx.edge_status.get());

    run_full(ctx, in, in.output_capacity, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    validate_output("parallel full scan", in, edge_ids, ctx.output_offsets.get(),
        ctx.output_items.get(), ctx.full_status.get());

    std::vector<double> posting_samples, concat_samples, bitmap_samples, full_samples;
    auto posting_call = [&] { run_postings(ctx, in, in.output_capacity, stream); };
    auto concat_call = [&] { run_concat(ctx, in, in.output_capacity, stream); };
    auto bitmap_call = [&] { run_bitmap(ctx, in, in.output_capacity, stream); };
    auto full_call = [&] { run_full(ctx, in, in.output_capacity, stream); };
    for (int i = 0; i < warmups; ++i) {
        if (measure_postings) posting_call();
        if (use_concat) concat_call();
        bitmap_call(); full_call();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int sample = 0; sample < sample_count; ++sample) {
        if (sample % 3 == 0) {
            if (measure_postings)
                posting_samples.push_back(timed_query(posting_call, start, stop, stream));
            if (use_concat)
                concat_samples.push_back(timed_query(concat_call, start, stop, stream));
            full_samples.push_back(timed_query(full_call, start, stop, stream));
            bitmap_samples.push_back(timed_query(bitmap_call, start, stop, stream));
        } else if (sample % 3 == 1) {
            full_samples.push_back(timed_query(full_call, start, stop, stream));
            if (use_concat)
                concat_samples.push_back(timed_query(concat_call, start, stop, stream));
            bitmap_samples.push_back(timed_query(bitmap_call, start, stop, stream));
            if (measure_postings)
                posting_samples.push_back(timed_query(posting_call, start, stop, stream));
        } else {
            bitmap_samples.push_back(timed_query(bitmap_call, start, stop, stream));
            if (use_concat)
                concat_samples.push_back(timed_query(concat_call, start, stop, stream));
            if (measure_postings)
                posting_samples.push_back(timed_query(posting_call, start, stop, stream));
            full_samples.push_back(timed_query(full_call, start, stop, stream));
        }
    }

    std::vector<std::vector<double>> bitmap_stages(6), full_stages(5), concat_stages(3);
    timed_stages_bitmap(ctx, in, bitmap_stages, stream);
    timed_stages_full(ctx, in, full_stages, stream);
    if (use_concat) timed_stages_concat(ctx, in, concat_stages, stream);
    if (overflow_probe) {
        test_all_overflows(ctx, in, measure_postings, use_concat, stream);
        if (use_concat) {
            run_concat(ctx, in, in.output_capacity, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            validate_output("count-scan-concat after overflow", in, edge_ids,
                ctx.output_offsets.get(), ctx.output_items.get(), ctx.concat_status.get());
        }
        run_bitmap(ctx, in, in.output_capacity, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        validate_output("CSR bitmap after overflow", in, edge_ids,
            ctx.output_offsets.get(), ctx.output_items.get(), ctx.edge_status.get());
    }

    std::cout << "{\"schema\":\"cellerator.csr_union_bitmap_v1\""
        << ",\"case_id\":\"" << name << "\""
        << ",\"edges\":" << ctx.edges << ",\"entities\":" << ctx.entities
        << ",\"cells\":" << ctx.cells << ",\"q_per_cell\":" << in.q_per_cell
        << ",\"query_terms_total\":" << in.query_entities.size()
        << ",\"candidate_required_total\":" << in.oracle_ordinals.size()
        << ",\"raw_posting_upper_bound\":" << in.output_capacity
        << ",\"correctness\":\"cpu_oracle_exact\""
        << ",\"device\":{\"name\":\"Tesla V100-SXM2-16GB\",\"uuid\":\"6c1cac7f-a360-0aef-ba98-2828bfd1db1a\",\"sm\":\"70\",\"runtime\":\"CUDA 12.9\"}"
        << ",\"overflow_check\":\""
        << (overflow_probe ? "exact_required_no_partial_write" : "not_run") << "\""
        << ",\"postings_status\":\""
        << (measure_postings ? "measured_exact" : "omitted_known_serial_high_q") << "\""
        << ",\"postings_operation_proxy_q_times_eligible\":"
        << serial_merge_work_proxy(in)
        << ",\"postings_overflow_check\":\""
        << (overflow_probe && measure_postings ? "exact_required_no_partial_write"
            : overflow_probe ? "omitted_known_serial_high_q" : "not_run") << "\""
        << ",\"count_scan_concat_status\":\""
        << (use_concat ? "measured_exact" : "not_applicable_order_or_duplicate_precondition") << "\""
        << ",\"query_upload_ms\":" << std::setprecision(9) << query_upload_ms
        << ",\"postings_p50_ms\":" << (measure_postings ? median(posting_samples) : -1.0)
        << ",\"postings_p10_ms\":" << (measure_postings ? percentile(posting_samples, 0.10) : -1.0)
        << ",\"postings_p90_ms\":" << (measure_postings ? percentile(posting_samples, 0.90) : -1.0)
        << ",\"postings_samples_ms\":"; print_samples(posting_samples);
    std::cout << ",\"fullscan_p50_ms\":" << median(full_samples)
        << ",\"fullscan_p10_ms\":" << percentile(full_samples, 0.10)
        << ",\"fullscan_p90_ms\":" << percentile(full_samples, 0.90)
        << ",\"fullscan_samples_ms\":"; print_samples(full_samples);
    std::cout << ",\"csr_bitmap_p50_ms\":" << median(bitmap_samples)
        << ",\"csr_bitmap_p10_ms\":" << percentile(bitmap_samples, 0.10)
        << ",\"csr_bitmap_p90_ms\":" << percentile(bitmap_samples, 0.90)
        << ",\"csr_bitmap_samples_ms\":"; print_samples(bitmap_samples);
    std::cout << ",\"count_scan_concat_p50_ms\":"
        << (use_concat ? median(concat_samples) : -1.0)
        << ",\"count_scan_concat_samples_ms\":";
    print_samples(concat_samples);
    std::cout << ",\"count_scan_concat_stage_p50_us\":{";
    if (use_concat) {
        const char *concat_stage_names[] = {"term_count", "scan_and_capacity", "scatter"};
        for (int i = 0; i < 3; ++i) {
            if (i) std::cout << ',';
            std::cout << '"' << concat_stage_names[i] << "\":"
                << median(concat_stages[i]);
        }
    }
    std::cout << "}";
    std::cout << ",\"csr_bitmap_stage_p50_us\":{";
    const char *bitmap_stage_names[] = {"zero", "posting_scatter", "popcount",
        "cub_scan", "capacity_finalize", "emit"};
    for (int i = 0; i < 6; ++i) {
        if (i) std::cout << ',';
        std::cout << '"' << bitmap_stage_names[i] << "\":" << median(bitmap_stages[i]);
    }
    std::cout << "},\"fullscan_stage_p50_us\":{";
    const char *full_stage_names[] = {"zero", "query_membership", "edge_count",
        "tile_scan_finalize", "emit"};
    for (int i = 0; i < 5; ++i) {
        if (i) std::cout << ',';
        std::cout << '"' << full_stage_names[i] << "\":" << median(full_stages[i]);
    }
    std::cout << "},\"memory_bytes\":{";
    const std::uint64_t common_offsets = ctx.output_offsets.bytes() + ctx.query_offsets.bytes();
    const std::uint64_t baseline_query_bytes = ctx.posting_queries.bytes();
    const std::uint64_t entity_query_bytes = ctx.query_entities.bytes();
    std::cout << "\"baseline_index\":" << ctx.baseline_index_bytes
        << ",\"baseline_scratch_ex_output\":" << ctx.baseline_scratch_bytes
        << ",\"baseline_total_resident\":" << ctx.baseline_index_bytes
            + ctx.baseline_scratch_bytes + ctx.output_items.bytes()
            + common_offsets + baseline_query_bytes
        << ",\"candidate_index\":" << ctx.candidate_index_bytes
        << ",\"candidate_scratch_ex_output\":" << ctx.candidate_scratch_bytes
        << ",\"candidate_total_resident\":" << ctx.candidate_index_bytes
            + ctx.candidate_scratch_bytes + ctx.output_items.bytes()
            + common_offsets + entity_query_bytes
        << ",\"count_scan_concat_index\":" << ctx.concat_index_bytes
        << ",\"count_scan_concat_scratch_ex_output\":" << ctx.concat_scratch_bytes
        << ",\"count_scan_concat_total_resident\":" << ctx.concat_index_bytes
            + ctx.concat_scratch_bytes + ctx.output_items.bytes()
            + common_offsets + entity_query_bytes
        << ",\"fullscan_index\":" << ctx.fullscan_index_bytes
        << ",\"fullscan_scratch_ex_output\":" << ctx.fullscan_scratch_bytes
        << ",\"fullscan_total_resident\":" << ctx.fullscan_index_bytes
            + ctx.fullscan_scratch_bytes + ctx.output_items.bytes()
            + common_offsets + entity_query_bytes
        << ",\"common_output\":" << ctx.output_items.bytes()
        << ",\"common_offsets\":" << common_offsets
        << ",\"baseline_posting_query_bytes\":" << baseline_query_bytes
        << ",\"candidate_entity_query_bytes\":" << entity_query_bytes
        << ",\"candidate_scan_temp_bytes\":" << ctx.edge_scan_temp_bytes
        << ",\"concat_scan_temp_bytes\":" << ctx.concat_scan_temp_bytes
        << ",\"fullscan_scan_temp_bytes\":" << ctx.tile_scan_temp_bytes
        << ",\"output_offsets\":" << ctx.output_offsets.bytes()
        << "}}" << std::endl;
}

} // namespace

int main(int argc, char **argv) {
    try {
        fs::path input_root;
        int warmups = 1;
        int samples = 9;
        bool cpu_preflight = false;
        std::vector<std::string> cases = {
            "q1_unique", "q32_unique", "q128_unique", "q1024_unique",
            "q2048_unique", "q32_duplicate4"};
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if ((arg == "--input-root" || arg == "--case") && i + 1 < argc) {
                if (arg == "--input-root") input_root = argv[++i];
                else cases = {argv[++i]};
            } else if (arg == "--warmups" && i + 1 < argc) {
                warmups = std::stoi(argv[++i]);
            } else if (arg == "--samples" && i + 1 < argc) {
                samples = std::stoi(argv[++i]);
            } else if (arg == "--cpu-preflight") {
                cpu_preflight = true;
            } else if (arg == "--help") {
                std::cout << "csr_union_bitmap_bench --input-root DIR [--case NAME] [--warmups N] [--samples N] [--cpu-preflight]\n";
                return 0;
            } else {
                throw std::runtime_error("invalid argument: " + arg);
            }
        }
        if (input_root.empty() || warmups < 0 || (!cpu_preflight && (samples < 3 || samples > 1000)))
            throw std::runtime_error("input root required, warmups must be nonnegative, timed samples must be in 3..1000");
        if (!fs::is_directory(input_root / "support"))
            throw std::runtime_error("fixture support directory missing");
        const auto setup_begin = clock_type::now();
        auto input = load_support(input_root);
        const auto direct_begin = clock_type::now();
        const auto direct = build_direct_csr(input.source_entities);
        const double direct_csr_ms = std::chrono::duration<double, std::milli>(
            clock_type::now() - direct_begin).count();
        assert_direct_csr_matches_fixture(direct, input);

        std::vector<support::logical_support_edge_v1> logical_edges;
        std::vector<support::logical_support_participant_v1> participants;
        ix::reverse_posting_view_v1 host_postings{};
        std::vector<ix::reverse_posting_record_v1> host_posting_entries;
        const auto posting_begin = clock_type::now();
        build_baseline_postings(input, logical_edges, participants,
            &host_postings, host_posting_entries);
        const double host_posting_ms = std::chrono::duration<double, std::milli>(
            clock_type::now() - posting_begin).count();
        const auto projection_begin = clock_type::now();
        const auto projection = build_cuda_projection(host_postings);
        const double host_projection_ms = std::chrono::duration<double, std::milli>(
            clock_type::now() - projection_begin).count();

        if (cpu_preflight) {
            validate_cpu_fixture(input_root, direct, input);
            std::cout << "{\"schema\":\"cellerator.csr_union_bitmap_cpu_preflight.v1\","
                << "\"support_edges\":" << kEdges << ",\"entity_count\":" << kEntities
                << ",\"cell_count\":" << kCells
                << ",\"direct_csr_matches_fixture\":true"
                << ",\"host_posting_projection\":true"
                << ",\"query_oracles_exact\":true"
                << ",\"independent_stable_id_order_oracle\":true"
                << ",\"concat_preconditions_checked\":true}\n";
            return 0;
        }

        assert_expected_uuid();

        const auto max_capacity = max_raw_capacity(input_root);
        std::uint64_t max_queries = kCells * 2048;
        benchmark_context ctx;
        ctx.query_capacity = max_queries;
        cudaStream_t stream{};
        CUDA_CHECK(cudaStreamCreate(&stream));
        const auto alloc_begin = clock_type::now();
        initialize_context(ctx, input, direct, projection, max_capacity, stream);
        const double cuda_setup_ms = std::chrono::duration<double, std::milli>(
            clock_type::now() - alloc_begin).count();
        const double setup_total_ms = std::chrono::duration<double, std::milli>(
            clock_type::now() - setup_begin).count();

        cudaEvent_t start{}, stop{};
        CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        for (const auto &name : cases) {
            const bool overflow_probe = name == "q2048_unique";
            run_case(ctx, input_root, name, warmups, samples,
                input, input.edge_ids, stream, start, stop, overflow_probe);
        }
        test_empty_query(ctx, load_case(input_root, "q1_unique"), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cerr << "{\"setup_ms\":{\"direct_csr_build\":" << std::setprecision(9)
            << direct_csr_ms << ",\"baseline_host_postings\":" << host_posting_ms
            << ",\"baseline_cuda_projection\":" << host_projection_ms
            << ",\"device_alloc_upload\":" << cuda_setup_ms
            << ",\"total\":" << setup_total_ms << "}}" << std::endl;
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "csr_union_bitmap_bench failed: " << error.what() << '\n';
        return 1;
    }
}
