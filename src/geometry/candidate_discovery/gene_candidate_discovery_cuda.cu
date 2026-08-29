/*
 * Validation/benchmark (2026-08-14, Cellerator 1ebb734): custom sm_70 CUDA
 * emits deterministic LSH records and bounded pairs while CUB provides stable
 * radix sorting, scans, and unique selection. CPU/GPU output is exact on the
 * focused runtime suite. ./build-sampling/geneCandidateDiscoveryBench on Tesla
 * V100, 65,536 cells x 30,000 genes, emitted 105,000 candidates with 100%
 * constructed-cluster recall; end-to-end minimum/median were 59.832/60.904 ms.
 */

#include <Cellerator/geometry/gene_candidate_discovery.hh>

#include <Cellerator/geometry/candidate_discovery/gene_candidate_hash.cuh>
#include "gene_candidate_internal.hh"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <limits>
#include <new>
#include <utility>

namespace cellerator::compute::gene_candidates {

namespace {

namespace cg = ::cellerator::compute::gene_support;
namespace ct = ::cellerator::types;
namespace cm = ::cellerator::memory;

struct device_buffers {
    cg::support_word_t *support = nullptr;
    std::uint64_t *global_rows = nullptr;
    ct::idx_t *all_genes = nullptr;
    ct::idx_t *nonempty_genes = nullptr;
    unsigned char *nonempty_flags = nullptr;
    std::uint64_t *sketches = nullptr;
    std::uint64_t *keys_a = nullptr;
    std::uint64_t *keys_b = nullptr;
    std::uint64_t *values_a = nullptr;
    std::uint64_t *values_b = nullptr;
    ct::u32 *heads = nullptr;
    ct::u32 *bucket_ids = nullptr;
    std::uint64_t *bucket_offsets = nullptr;
    std::uint64_t *pair_counts = nullptr;
    std::uint64_t *pair_offsets = nullptr;
    std::uint64_t *pairs_a = nullptr;
    std::uint64_t *pairs_b = nullptr;
    std::uint64_t *oversized_buckets = nullptr;
    std::uint64_t *discarded_members = nullptr;
    int *unique_count = nullptr;
    void *cub_temp = nullptr;
};

cudaError_t release_pointer(void *pointer, cudaError_t current) {
    if (pointer == nullptr) return current;
    const cudaError_t status = cudaFree(pointer);
    return current == cudaSuccess ? status : current;
}

cudaError_t free_device_buffers(device_buffers *buffers) {
    if (buffers == nullptr) return cudaSuccess;
    cudaError_t status = cudaSuccess;
    status = release_pointer(buffers->cub_temp, status);
    status = release_pointer(buffers->unique_count, status);
    status = release_pointer(buffers->discarded_members, status);
    status = release_pointer(buffers->oversized_buckets, status);
    status = release_pointer(buffers->pairs_b, status);
    status = release_pointer(buffers->pairs_a, status);
    status = release_pointer(buffers->pair_offsets, status);
    status = release_pointer(buffers->pair_counts, status);
    status = release_pointer(buffers->bucket_offsets, status);
    status = release_pointer(buffers->bucket_ids, status);
    status = release_pointer(buffers->heads, status);
    status = release_pointer(buffers->values_b, status);
    status = release_pointer(buffers->values_a, status);
    status = release_pointer(buffers->keys_b, status);
    status = release_pointer(buffers->keys_a, status);
    status = release_pointer(buffers->sketches, status);
    status = release_pointer(buffers->nonempty_genes, status);
    status = release_pointer(buffers->nonempty_flags, status);
    status = release_pointer(buffers->all_genes, status);
    status = release_pointer(buffers->global_rows, status);
    status = release_pointer(buffers->support, status);
    *buffers = {};
    return status;
}

bool cuda_ok(cudaError_t status, const char *operation, std::string *error) {
    if (status == cudaSuccess) return true;
    detail::set_error(error, std::string(operation) + ": " + cudaGetErrorString(status));
    return false;
}

__global__ void emit_lsh_records_kernel(const std::uint64_t *sketches,
                                        const ct::idx_t *nonempty_genes,
                                        std::uint64_t nonempty_gene_count,
                                        std::uint32_t sketch_count,
                                        std::uint32_t rows_per_band,
                                        std::uint64_t seed,
                                        std::uint64_t record_count,
                                        std::uint64_t *keys,
                                        std::uint64_t *values) {
    const std::uint64_t record = (std::uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (record >= record_count) return;
    const std::uint32_t band = (std::uint32_t) (record / nonempty_gene_count);
    const std::uint64_t gene_position = record % nonempty_gene_count;
    const std::uint64_t *band_values = sketches + gene_position * sketch_count
        + (std::size_t) band * rows_per_band;
    keys[record] = detail::lsh_band_key_v1(band_values, rows_per_band, seed, band);
    values[record] = ((std::uint64_t) band << 32u) | nonempty_genes[gene_position];
}

__global__ void mark_bucket_heads_kernel(const std::uint64_t *keys,
                                         const std::uint64_t *values,
                                         std::uint64_t record_count,
                                         ct::u32 *heads) {
    const std::uint64_t record = (std::uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (record >= record_count) return;
    const std::uint32_t band = (std::uint32_t) (values[record] >> 32u);
    heads[record] = record == 0u || keys[record] != keys[record - 1u]
        || band != (std::uint32_t) (values[record - 1u] >> 32u);
}

__global__ void scatter_bucket_offsets_kernel(const ct::u32 *heads,
                                              const ct::u32 *bucket_ids,
                                              std::uint64_t record_count,
                                              std::uint64_t *bucket_offsets) {
    const std::uint64_t record = (std::uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (record >= record_count) return;
    if (heads[record] != 0u) bucket_offsets[bucket_ids[record]] = record;
    if (record + 1u == record_count) {
        bucket_offsets[(std::size_t) bucket_ids[record] + heads[record]] = record_count;
    }
}

__global__ void count_bucket_pairs_kernel(const std::uint64_t *bucket_offsets,
                                          std::uint64_t bucket_count,
                                          std::uint32_t maximum_bucket_size,
                                          std::uint64_t *pair_counts,
                                          std::uint64_t *oversized_buckets,
                                          std::uint64_t *discarded_members) {
    const std::uint64_t bucket = (std::uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket >= bucket_count) return;
    const std::uint64_t bucket_size = bucket_offsets[bucket + 1u] - bucket_offsets[bucket];
    const std::uint64_t selected = min(bucket_size, (std::uint64_t) maximum_bucket_size);
    pair_counts[bucket] = selected < 2u ? 0u : selected * (selected - 1u) / 2u;
    if (bucket_size > maximum_bucket_size) {
        atomicAdd((unsigned long long *) oversized_buckets, 1ull);
        atomicAdd((unsigned long long *) discarded_members,
                  (unsigned long long) (bucket_size - maximum_bucket_size));
    }
}

__global__ void emit_bucket_pairs_kernel(const std::uint64_t *keys,
                                         const std::uint64_t *values,
                                         const std::uint64_t *bucket_offsets,
                                         const std::uint64_t *pair_offsets,
                                         std::uint64_t bucket_count,
                                         std::uint32_t maximum_bucket_size,
                                         std::uint64_t seed,
                                         std::uint64_t *pairs) {
    const std::uint64_t bucket = blockIdx.x;
    if (bucket >= bucket_count) return;
    const std::uint64_t begin = bucket_offsets[bucket];
    const std::uint64_t bucket_size = bucket_offsets[bucket + 1u] - begin;
    const std::uint64_t selected = min(bucket_size, (std::uint64_t) maximum_bucket_size);
    if (selected < 2u) return;
    const std::uint32_t band = (std::uint32_t) (values[begin] >> 32u);
    const std::uint64_t start = bucket_size > maximum_bucket_size
        ? detail::oversized_bucket_window_start_v1(keys[begin], seed, band, bucket_size)
        : 0u;
    for (std::uint64_t i = threadIdx.x; i < selected; i += blockDim.x) {
        const ct::idx_t gene_i = (ct::idx_t) values[begin + (start + i) % bucket_size];
        const std::uint64_t prefix = i * (2u * selected - i - 1u) / 2u;
        for (std::uint64_t j = i + 1u; j < selected; ++j) {
            const ct::idx_t gene_j = (ct::idx_t) values[begin + (start + j) % bucket_size];
            const ct::idx_t gene_a = min(gene_i, gene_j), gene_b = max(gene_i, gene_j);
            pairs[pair_offsets[bucket] + prefix + j - i - 1u] =
                ((std::uint64_t) gene_a << 32u) | gene_b;
        }
    }
}

__global__ void unpack_candidate_pairs_kernel(std::uint64_t *encoded,
                                              std::uint64_t count) {
    const std::uint64_t index = (std::uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::uint64_t key = encoded[index];
    gene_candidate_pair *pairs = (gene_candidate_pair *) encoded;
    pairs[index].gene_a = (ct::idx_t) (key >> 32u);
    pairs[index].gene_b = (ct::idx_t) key;
}

__global__ void mark_nonempty_genes_kernel(
    const ct::count_value_t *counts,
    std::uint64_t gene_count,
    ct::idx_t *genes,
    unsigned char *flags) {
    const std::uint64_t gene = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (gene >= gene_count) return;
    genes[gene] = static_cast<ct::idx_t>(gene);
    flags[gene] = counts[gene] != 0u ? 1u : 0u;
}

constexpr std::size_t prepared_alignment = 256u;

bool checked_align_add(std::size_t current,
                       std::size_t bytes,
                       std::size_t alignment,
                       std::size_t *out) {
    if (out == nullptr || alignment == 0u || (alignment & (alignment - 1u)) != 0u) return false;
    const std::size_t mask = alignment - 1u;
    if (current > std::numeric_limits<std::size_t>::max() - mask) return false;
    const std::size_t aligned = (current + mask) & ~mask;
    if (bytes > std::numeric_limits<std::size_t>::max() - aligned) return false;
    *out = aligned + bytes;
    return true;
}

template<class T>
bool add_array_bytes(std::size_t count, std::size_t *bytes) {
    std::size_t array_bytes = 0u;
    return detail::checked_multiply(count, sizeof(T), &array_bytes)
        && checked_align_add(*bytes, array_bytes, alignof(T), bytes);
}

bool bind_prepared_buffers(
    const cg::gene_support_layout &layout,
    const candidate_discovery_config &config,
    const candidate_discovery_device_requirements &requirements,
    cm::workspace *workspace,
    device_buffers *buffers,
    std::string *error) {
    const std::size_t gene_count = static_cast<std::size_t>(layout.gene_count);
    const std::size_t record_count = static_cast<std::size_t>(requirements.bounds.maximum_lsh_records);
    const std::size_t pair_capacity = static_cast<std::size_t>(requirements.candidate_capacity);
    auto take = [&](auto **pointer, std::size_t count) {
        return cm::take(workspace, count, pointer) == cm::status::success;
    };
    if (!take(&buffers->all_genes, gene_count)
        || !take(&buffers->nonempty_genes, gene_count)
        || !take(&buffers->nonempty_flags, gene_count)
        || !take(&buffers->unique_count, 1u)
        || !take(&buffers->sketches, gene_count * config.sketch_count)
        || !take(&buffers->keys_a, record_count)
        || !take(&buffers->keys_b, record_count)
        || !take(&buffers->values_a, record_count)
        || !take(&buffers->values_b, record_count)
        || !take(&buffers->heads, record_count)
        || !take(&buffers->bucket_ids, record_count)
        || !take(&buffers->bucket_offsets, record_count + 1u)
        || !take(&buffers->pair_counts, record_count)
        || !take(&buffers->pair_offsets, record_count + 1u)
        || !take(&buffers->oversized_buckets, 1u)
        || !take(&buffers->discarded_members, 1u)
        || !take(&buffers->pairs_a, pair_capacity)
        || !take(&buffers->pairs_b, pair_capacity)
        || cm::take_bytes(workspace, requirements.cub_temporary_bytes,
                          prepared_alignment, &buffers->cub_temp) != cm::status::success) {
        detail::set_error(error, "prepared candidate workspace capacity is insufficient");
        return false;
    }
    return true;
}

} // namespace

bool prepare_candidate_discovery_cuda(
    const cg::gene_support_layout &support_layout,
    const candidate_discovery_config &config,
    int device,
    candidate_discovery_device_requirements *out,
    std::string *error) {
    if (out == nullptr) {
        detail::set_error(error, "candidate device requirements output is null");
        return false;
    }
    candidate_discovery_device_requirements result;
    if (device < 0 || !calculate_candidate_discovery_bounds(
            support_layout, config, &result.bounds, error)) return false;
    if (result.bounds.maximum_lsh_records > static_cast<std::uint64_t>(INT_MAX)
        || result.bounds.maximum_raw_pair_occurrences > static_cast<std::uint64_t>(INT_MAX)) {
        detail::set_error(error, "candidate preflight bound exceeds CUB item range");
        return false;
    }
    result.candidate_capacity = result.bounds.maximum_raw_pair_occurrences;
    const int genes = static_cast<int>(support_layout.gene_count);
    const int records = static_cast<int>(result.bounds.maximum_lsh_records);
    const int pairs = static_cast<int>(result.candidate_capacity);
    std::size_t select_temp = 0u, sort_record_temp = 0u, scan_record_temp = 0u;
    std::size_t scan_pair_offset_temp = 0u;
    std::size_t sort_pair_temp = 0u, unique_temp = 0u;
    cudaError_t status = cub::DeviceSelect::Flagged(
        nullptr, select_temp, static_cast<ct::idx_t *>(nullptr),
        static_cast<unsigned char *>(nullptr), static_cast<ct::idx_t *>(nullptr),
        static_cast<int *>(nullptr), genes);
    if (status == cudaSuccess) status = cub::DeviceRadixSort::SortPairs(
        nullptr, sort_record_temp, static_cast<std::uint64_t *>(nullptr),
        static_cast<std::uint64_t *>(nullptr), static_cast<std::uint64_t *>(nullptr),
        static_cast<std::uint64_t *>(nullptr), records);
    if (status == cudaSuccess) status = cub::DeviceScan::ExclusiveSum(
        nullptr, scan_record_temp, static_cast<ct::u32 *>(nullptr),
        static_cast<ct::u32 *>(nullptr), records);
    if (status == cudaSuccess) status = cub::DeviceScan::ExclusiveSum(
        nullptr, scan_pair_offset_temp, static_cast<std::uint64_t *>(nullptr),
        static_cast<std::uint64_t *>(nullptr), records);
    if (status == cudaSuccess) status = cub::DeviceRadixSort::SortKeys(
        nullptr, sort_pair_temp, static_cast<std::uint64_t *>(nullptr),
        static_cast<std::uint64_t *>(nullptr), pairs);
    if (status == cudaSuccess) status = cub::DeviceSelect::Unique(
        nullptr, unique_temp, static_cast<std::uint64_t *>(nullptr),
        static_cast<std::uint64_t *>(nullptr), static_cast<int *>(nullptr), pairs);
    if (status != cudaSuccess) return cuda_ok(status, "candidate CUB scratch query", error);
    result.cub_temporary_bytes = std::max(
        {select_temp, sort_record_temp, scan_record_temp, scan_pair_offset_temp,
         sort_pair_temp, unique_temp});

    std::size_t bytes = 0u;
    const std::size_t gene_count = static_cast<std::size_t>(support_layout.gene_count);
    const std::size_t record_count = static_cast<std::size_t>(result.bounds.maximum_lsh_records);
    const std::size_t pair_capacity = static_cast<std::size_t>(result.candidate_capacity);
    if (!add_array_bytes<ct::idx_t>(gene_count, &bytes)
        || !add_array_bytes<ct::idx_t>(gene_count, &bytes)
        || !add_array_bytes<unsigned char>(gene_count, &bytes)
        || !add_array_bytes<int>(1u, &bytes)
        || !add_array_bytes<std::uint64_t>(gene_count * config.sketch_count, &bytes)
        || !add_array_bytes<std::uint64_t>(record_count, &bytes)
        || !add_array_bytes<std::uint64_t>(record_count, &bytes)
        || !add_array_bytes<std::uint64_t>(record_count, &bytes)
        || !add_array_bytes<std::uint64_t>(record_count, &bytes)
        || !add_array_bytes<ct::u32>(record_count, &bytes)
        || !add_array_bytes<ct::u32>(record_count, &bytes)
        || !add_array_bytes<std::uint64_t>(record_count + 1u, &bytes)
        || !add_array_bytes<std::uint64_t>(record_count, &bytes)
        || !add_array_bytes<std::uint64_t>(record_count + 1u, &bytes)
        || !add_array_bytes<std::uint64_t>(1u, &bytes)
        || !add_array_bytes<std::uint64_t>(1u, &bytes)
        || !add_array_bytes<std::uint64_t>(pair_capacity, &bytes)
        || !add_array_bytes<std::uint64_t>(pair_capacity, &bytes)
        || !checked_align_add(bytes, result.cub_temporary_bytes, prepared_alignment, &bytes)) {
        detail::set_error(error, "candidate prepared workspace size overflows size_t");
        return false;
    }
    result.workspace = {bytes, static_cast<std::uint32_t>(prepared_alignment),
        {cm::domain::device, static_cast<std::int16_t>(device), -1, 0u}};
    *out = result;
    return true;
}

bool discover_gene_candidates_cuda_prepared(
    const cg::gene_support_device_view &support,
    const candidate_discovery_config &config,
    const candidate_discovery_device_requirements &requirements,
    cm::workspace workspace,
    cudaStream_t stream,
    gene_candidate_pair_device_view *out,
    std::string *error) {
    static_assert(sizeof(gene_candidate_pair) == sizeof(std::uint64_t),
                  "candidate pairs must remain one packed 64-bit relation");
    if (out == nullptr || support.provenance == nullptr) {
        detail::set_error(error, "prepared candidate output or support provenance is null");
        return false;
    }
    candidate_discovery_bounds expected;
    if (!detail::validate_config(config, error)
        || !calculate_candidate_discovery_bounds(support.layout, config, &expected, error)
        || expected.maximum_lsh_records != requirements.bounds.maximum_lsh_records
        || expected.maximum_raw_pair_occurrences != requirements.candidate_capacity) {
        detail::set_error(error, "prepared candidate requirements do not match support/configuration");
        return false;
    }
    const int device = workspace.where.device_ordinal;
    const cm::placement expected_where{cm::domain::device,
        static_cast<std::int16_t>(device), -1, 0u};
    if (workspace.where != expected_where
        || support.gene_support.where != expected_where
        || support.detected_cell_counts.where != expected_where
        || support.sampled_position_to_global_row.where != expected_where
        || workspace.bytes < requirements.workspace.bytes) {
        detail::set_error(error, "prepared candidate views must share the preflight CUDA placement");
        return false;
    }
    device_buffers buffers;
    if (!bind_prepared_buffers(support.layout, config, requirements,
                               &workspace, &buffers, error)) return false;
    bool ok = true;
    auto step = [&](cudaError_t status, const char *operation) {
        if (ok && status != cudaSuccess) ok = cuda_ok(status, operation, error);
    };
    constexpr unsigned int threads = 256u;
    const std::uint64_t gene_count = support.layout.gene_count;
    if (gene_count != 0u) {
        const unsigned int blocks = static_cast<unsigned int>((gene_count + threads - 1u) / threads);
        mark_nonempty_genes_kernel<<<blocks, threads, 0u, stream>>>(
            support.detected_cell_counts.data, gene_count,
            buffers.all_genes, buffers.nonempty_flags);
        step(cudaGetLastError(), "mark nonempty genes launch");
    }
    std::size_t select_temp = requirements.cub_temporary_bytes;
    if (ok) step(cub::DeviceSelect::Flagged(
        buffers.cub_temp, select_temp, buffers.all_genes, buffers.nonempty_flags,
        buffers.nonempty_genes, buffers.unique_count, static_cast<int>(gene_count), stream),
        "CUB nonempty-gene selection");
    int nonempty_int = 0;
    ct::u32 support_error = 0u;
    if (ok) step(cudaMemcpyAsync(&nonempty_int, buffers.unique_count, sizeof(nonempty_int),
                                 cudaMemcpyDeviceToHost, stream),
                 "copy nonempty-gene count");
    if (ok) step(cudaMemcpyAsync(&support_error, support.device_error, sizeof(support_error),
                                 cudaMemcpyDeviceToHost, stream),
                 "copy support validation status");
    if (ok) step(cudaStreamSynchronize(stream), "synchronize support/candidate boundary");
    if (!ok) return false;
    if (support_error != 0u) {
        detail::set_error(error, "device-resident gene support rejected sampled CSR input");
        return false;
    }
    if (nonempty_int < 0) {
        detail::set_error(error, "device nonempty-gene count is negative");
        return false;
    }
    const std::uint64_t nonempty_count = static_cast<std::uint64_t>(nonempty_int);
    candidate_discovery_device_evidence evidence;
    evidence.config = config;
    evidence.sampling = support.provenance;
    evidence.sampled_cell_count = support.layout.sampled_cell_count;
    evidence.gene_count = support.layout.gene_count;
    evidence.nonempty_gene_count = nonempty_count;
    evidence.device_cub_temporary_bytes = requirements.cub_temporary_bytes;
    evidence.device_peak_bytes = support.layout.support_bytes + requirements.workspace.bytes;
    if (nonempty_count < 2u) {
        *out = {{reinterpret_cast<gene_candidate_pair *>(buffers.pairs_a), 0u, workspace.where},
                0u, requirements.candidate_capacity, evidence};
        return true;
    }

    const std::uint64_t record_count = nonempty_count * config.lsh_bands;
    evidence.lsh_record_count = record_count;
    step(detail::launch_gene_minhash(
        support.gene_support.data, support.sampled_position_to_global_row.data,
        buffers.nonempty_genes, nonempty_count, support.layout.words_per_gene,
        config.sketch_count, config.seed, buffers.sketches, stream),
        "gene MinHash launch");
    const unsigned int record_blocks = static_cast<unsigned int>((record_count + threads - 1u) / threads);
    if (ok) {
        emit_lsh_records_kernel<<<record_blocks, threads, 0u, stream>>>(
            buffers.sketches, buffers.nonempty_genes, nonempty_count, config.sketch_count,
            config.rows_per_band, config.seed, record_count, buffers.keys_a, buffers.values_a);
        step(cudaGetLastError(), "emit prepared LSH records launch");
    }
    std::size_t sort_record_temp = requirements.cub_temporary_bytes;
    if (ok) step(cub::DeviceRadixSort::SortPairs(
        buffers.cub_temp, sort_record_temp, buffers.keys_a, buffers.keys_b,
        buffers.values_a, buffers.values_b, static_cast<int>(record_count), 0, 64, stream),
        "CUB prepared LSH radix sort");
    if (ok) {
        mark_bucket_heads_kernel<<<record_blocks, threads, 0u, stream>>>(
            buffers.keys_b, buffers.values_b, record_count, buffers.heads);
        step(cudaGetLastError(), "mark prepared LSH bucket heads launch");
    }
    std::size_t scan_head_temp = requirements.cub_temporary_bytes;
    if (ok) step(cub::DeviceScan::ExclusiveSum(
        buffers.cub_temp, scan_head_temp, buffers.heads, buffers.bucket_ids,
        static_cast<int>(record_count), stream), "CUB prepared bucket scan");
    ct::u32 last_bucket_id = 0u, last_head = 0u;
    if (ok) step(cudaMemcpyAsync(&last_bucket_id, buffers.bucket_ids + record_count - 1u,
                                 sizeof(last_bucket_id), cudaMemcpyDeviceToHost, stream),
                 "copy prepared last bucket id");
    if (ok) step(cudaMemcpyAsync(&last_head, buffers.heads + record_count - 1u,
                                 sizeof(last_head), cudaMemcpyDeviceToHost, stream),
                 "copy prepared last bucket head");
    if (ok) step(cudaStreamSynchronize(stream), "synchronize prepared bucket count");
    if (!ok) return false;
    const std::uint64_t bucket_count = static_cast<std::uint64_t>(last_bucket_id) + last_head;
    evidence.bucket_count = bucket_count;
    scatter_bucket_offsets_kernel<<<record_blocks, threads, 0u, stream>>>(
        buffers.heads, buffers.bucket_ids, record_count, buffers.bucket_offsets);
    step(cudaGetLastError(), "scatter prepared bucket offsets launch");
    const unsigned int bucket_blocks = static_cast<unsigned int>((bucket_count + threads - 1u) / threads);
    if (ok) step(cudaMemsetAsync(buffers.oversized_buckets, 0, sizeof(std::uint64_t), stream),
                 "clear prepared oversized-bucket count");
    if (ok) step(cudaMemsetAsync(buffers.discarded_members, 0, sizeof(std::uint64_t), stream),
                 "clear prepared discarded-member count");
    if (ok && bucket_count != 0u) {
        count_bucket_pairs_kernel<<<bucket_blocks, threads, 0u, stream>>>(
            buffers.bucket_offsets, bucket_count, config.maximum_bucket_size,
            buffers.pair_counts, buffers.oversized_buckets, buffers.discarded_members);
        step(cudaGetLastError(), "count prepared bucket pairs launch");
    }
    std::size_t scan_pair_temp = requirements.cub_temporary_bytes;
    if (ok) step(cub::DeviceScan::ExclusiveSum(
        buffers.cub_temp, scan_pair_temp, buffers.pair_counts, buffers.pair_offsets,
        static_cast<int>(bucket_count), stream), "CUB prepared pair-offset scan");
    std::uint64_t last_pair_offset = 0u, last_pair_count = 0u;
    if (ok && bucket_count != 0u) {
        step(cudaMemcpyAsync(&last_pair_offset, buffers.pair_offsets + bucket_count - 1u,
                             sizeof(last_pair_offset), cudaMemcpyDeviceToHost, stream),
             "copy prepared last pair offset");
        step(cudaMemcpyAsync(&last_pair_count, buffers.pair_counts + bucket_count - 1u,
                             sizeof(last_pair_count), cudaMemcpyDeviceToHost, stream),
             "copy prepared last pair count");
    }
    if (ok) step(cudaStreamSynchronize(stream), "synchronize prepared raw-pair count");
    if (!ok) return false;
    const std::uint64_t raw_pair_count = last_pair_offset + last_pair_count;
    evidence.raw_pair_occurrences = raw_pair_count;
    if (raw_pair_count > requirements.candidate_capacity) {
        detail::set_error(error, "prepared candidate raw pairs exceeded preflight capacity");
        return false;
    }
    std::uint64_t unique_pair_count = 0u;
    if (raw_pair_count != 0u) {
        emit_bucket_pairs_kernel<<<static_cast<unsigned int>(bucket_count), threads, 0u, stream>>>(
            buffers.keys_b, buffers.values_b, buffers.bucket_offsets, buffers.pair_offsets,
            bucket_count, config.maximum_bucket_size, config.seed, buffers.pairs_a);
        step(cudaGetLastError(), "emit prepared candidate pairs launch");
        std::size_t sort_pair_temp = requirements.cub_temporary_bytes;
        if (ok) step(cub::DeviceRadixSort::SortKeys(
            buffers.cub_temp, sort_pair_temp, buffers.pairs_a, buffers.pairs_b,
            static_cast<int>(raw_pair_count), 0, 64, stream),
            "CUB prepared candidate sort");
        std::size_t unique_temp = requirements.cub_temporary_bytes;
        if (ok) step(cub::DeviceSelect::Unique(
            buffers.cub_temp, unique_temp, buffers.pairs_b, buffers.pairs_a,
            buffers.unique_count, static_cast<int>(raw_pair_count), stream),
            "CUB prepared candidate unique");
    } else {
        step(cudaMemsetAsync(buffers.unique_count, 0, sizeof(int), stream),
             "clear prepared unique count");
    }
    int unique_int = 0;
    std::uint64_t oversized = 0u, discarded = 0u;
    if (ok) step(cudaMemcpyAsync(&unique_int, buffers.unique_count, sizeof(unique_int),
                                 cudaMemcpyDeviceToHost, stream), "copy prepared unique count");
    if (ok) step(cudaMemcpyAsync(&oversized, buffers.oversized_buckets, sizeof(oversized),
                                 cudaMemcpyDeviceToHost, stream), "copy oversized-bucket evidence");
    if (ok) step(cudaMemcpyAsync(&discarded, buffers.discarded_members, sizeof(discarded),
                                 cudaMemcpyDeviceToHost, stream), "copy discarded-member evidence");
    if (ok) step(cudaStreamSynchronize(stream), "synchronize prepared candidate evidence");
    if (!ok || unique_int < 0) {
        if (ok) detail::set_error(error, "prepared candidate unique count is negative");
        return false;
    }
    unique_pair_count = static_cast<std::uint64_t>(unique_int);
    evidence.oversized_bucket_count = oversized;
    evidence.discarded_bucket_members = discarded;
    evidence.unique_candidate_count = unique_pair_count;
    if (unique_pair_count != 0u) {
        const unsigned int blocks = static_cast<unsigned int>((unique_pair_count + threads - 1u) / threads);
        unpack_candidate_pairs_kernel<<<blocks, threads, 0u, stream>>>(buffers.pairs_a, unique_pair_count);
        step(cudaGetLastError(), "unpack prepared candidate pairs launch");
    }
    if (!ok) return false;
    *out = {{reinterpret_cast<gene_candidate_pair *>(buffers.pairs_a),
            static_cast<std::size_t>(unique_pair_count), workspace.where},
            unique_pair_count, requirements.candidate_capacity, evidence};
    return true;
}

bool materialize_gene_candidates_host(
    const gene_candidate_pair_device_view &device_candidates,
    cudaStream_t stream,
    owned_gene_candidates *out,
    std::string *error) {
    if (out == nullptr || device_candidates.count > device_candidates.capacity
        || device_candidates.pairs.count < device_candidates.count
        || device_candidates.pairs.where.kind != cm::domain::device) {
        detail::set_error(error, "terminal candidate materialization contract is invalid");
        return false;
    }
    std::unique_ptr<gene_candidate_pair[]> pairs;
    if (device_candidates.count != 0u) {
        pairs.reset(new (std::nothrow) gene_candidate_pair[
            static_cast<std::size_t>(device_candidates.count)]);
        if (pairs == nullptr) {
            detail::set_error(error, "failed to allocate terminal candidate output");
            return false;
        }
        cudaError_t status = cudaMemcpyAsync(
            pairs.get(), device_candidates.pairs.data,
            static_cast<std::size_t>(device_candidates.count) * sizeof(gene_candidate_pair),
            cudaMemcpyDeviceToHost, stream);
        if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) return cuda_ok(status, "terminal candidate materialization", error);
    }
    if (device_candidates.evidence.sampling == nullptr) {
        detail::set_error(error, "terminal candidate evidence has null sampling provenance");
        return false;
    }
    try {
        candidate_discovery_provenance provenance;
        provenance.algorithm = candidate_algorithm_name;
        provenance.hash_version = candidate_hash_version;
        provenance.config = device_candidates.evidence.config;
        provenance.sampling = *device_candidates.evidence.sampling;
        provenance.sampled_cell_count = device_candidates.evidence.sampled_cell_count;
        provenance.gene_count = device_candidates.evidence.gene_count;
        provenance.nonempty_gene_count = device_candidates.evidence.nonempty_gene_count;
        provenance.lsh_record_count = device_candidates.evidence.lsh_record_count;
        provenance.bucket_count = device_candidates.evidence.bucket_count;
        provenance.oversized_bucket_count = device_candidates.evidence.oversized_bucket_count;
        provenance.discarded_bucket_members = device_candidates.evidence.discarded_bucket_members;
        provenance.raw_pair_occurrences = device_candidates.evidence.raw_pair_occurrences;
        provenance.unique_candidate_count = device_candidates.evidence.unique_candidate_count;
        provenance.device_cub_temporary_bytes = device_candidates.evidence.device_cub_temporary_bytes;
        provenance.device_peak_bytes = device_candidates.evidence.device_peak_bytes;
        provenance.bucket_cap_rule = candidate_bucket_cap_rule;
        *out = owned_gene_candidates(std::move(pairs), device_candidates.count,
                                     std::move(provenance));
    } catch (const std::bad_alloc &) {
        detail::set_error(error, "failed to copy terminal candidate provenance");
        return false;
    }
    return true;
}

bool discover_gene_candidates_cuda(const cg::gene_support_bitset_view &support,
                                   const candidate_discovery_config &config,
                                   int device,
                                   owned_gene_candidates *out,
                                   std::string *error) {
    static_assert(sizeof(gene_candidate_pair) == sizeof(std::uint64_t),
                  "candidate pairs must remain one packed 64-bit relation");
    if (out == nullptr) {
        detail::set_error(error, "owned candidate output is null");
        return false;
    }
    candidate_discovery_bounds bounds;
    if (!detail::validate_config(config, error)
        || !detail::validate_support_view(support, error)
        || !calculate_candidate_discovery_bounds(support.layout, config, &bounds, error)) {
        return false;
    }
    std::unique_ptr<ct::idx_t[]> nonempty_genes;
    std::uint64_t nonempty_count = 0u;
    if (!detail::collect_nonempty_genes(support, &nonempty_genes, &nonempty_count, error)) return false;
    candidate_discovery_provenance provenance =
        detail::make_provenance(support, config, nonempty_count);
    if (nonempty_count < 2u) {
        *out = owned_gene_candidates(nullptr, 0u, std::move(provenance));
        return true;
    }

    const std::uint64_t record_count = nonempty_count * config.lsh_bands;
    if (record_count > (std::uint64_t) std::numeric_limits<int>::max()) {
        detail::set_error(error, "actual candidate LSH record count exceeds CUB item range");
        return false;
    }
    const std::size_t support_bytes = support.layout.support_bytes;
    const std::size_t mapping_bytes = (std::size_t) support.layout.sampled_cell_count * sizeof(std::uint64_t);
    const std::size_t nonempty_bytes = (std::size_t) nonempty_count * sizeof(ct::idx_t);
    const std::size_t sketch_bytes = (std::size_t) nonempty_count * config.sketch_count * sizeof(std::uint64_t);
    const std::size_t record_u64_bytes = (std::size_t) record_count * sizeof(std::uint64_t);
    const std::size_t record_u32_bytes = (std::size_t) record_count * sizeof(ct::u32);
    const std::size_t record_offset_bytes = ((std::size_t) record_count + 1u) * sizeof(std::uint64_t);

    int device_count = 0, previous_device = 0;
    if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", error)) return false;
    if (device < 0 || device >= device_count) {
        detail::set_error(error, "requested CUDA device is unavailable");
        return false;
    }
    if (!cuda_ok(cudaGetDevice(&previous_device), "cudaGetDevice", error)
        || !cuda_ok(cudaSetDevice(device), "cudaSetDevice", error)) return false;

    device_buffers buffers;
    bool ok = true;
    auto cuda_step = [&](cudaError_t status, const char *operation) {
        if (ok && status != cudaSuccess) ok = cuda_ok(status, operation, error);
    };
    auto allocate = [&](void **pointer, std::size_t bytes, const char *operation) {
        if (ok && bytes != 0u) cuda_step(cudaMalloc(pointer, bytes), operation);
    };

    allocate((void **) &buffers.support, support_bytes, "cudaMalloc(candidate support)");
    allocate((void **) &buffers.global_rows, mapping_bytes, "cudaMalloc(candidate global rows)");
    allocate((void **) &buffers.nonempty_genes, nonempty_bytes, "cudaMalloc(nonempty genes)");
    allocate((void **) &buffers.sketches, sketch_bytes, "cudaMalloc(MinHash sketches)");
    allocate((void **) &buffers.keys_a, record_u64_bytes, "cudaMalloc(LSH keys A)");
    allocate((void **) &buffers.keys_b, record_u64_bytes, "cudaMalloc(LSH keys B)");
    allocate((void **) &buffers.values_a, record_u64_bytes, "cudaMalloc(LSH values A)");
    allocate((void **) &buffers.values_b, record_u64_bytes, "cudaMalloc(LSH values B)");
    allocate((void **) &buffers.heads, record_u32_bytes, "cudaMalloc(bucket heads)");
    allocate((void **) &buffers.bucket_ids, record_u32_bytes, "cudaMalloc(bucket ids)");
    allocate((void **) &buffers.bucket_offsets, record_offset_bytes, "cudaMalloc(bucket offsets)");
    allocate((void **) &buffers.pair_counts, record_u64_bytes, "cudaMalloc(bucket pair counts)");
    allocate((void **) &buffers.pair_offsets, record_offset_bytes, "cudaMalloc(bucket pair offsets)");
    allocate((void **) &buffers.oversized_buckets, sizeof(std::uint64_t), "cudaMalloc(oversized buckets)");
    allocate((void **) &buffers.discarded_members, sizeof(std::uint64_t), "cudaMalloc(discarded members)");
    allocate((void **) &buffers.unique_count, sizeof(int), "cudaMalloc(unique count)");

    if (ok && support_bytes != 0u) {
        cuda_step(cudaMemcpy(buffers.support, support.gene_support, support_bytes,
                             cudaMemcpyHostToDevice), "cudaMemcpy(candidate support H2D)");
    }
    if (ok && mapping_bytes != 0u) {
        cuda_step(cudaMemcpy(buffers.global_rows, support.sampled_position_to_global_row,
                             mapping_bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy(candidate global rows H2D)");
    }
    if (ok) cuda_step(cudaMemcpy(buffers.nonempty_genes, nonempty_genes.get(), nonempty_bytes,
                                 cudaMemcpyHostToDevice), "cudaMemcpy(nonempty genes H2D)");
    if (ok) cuda_step(cudaMemset(buffers.oversized_buckets, 0, sizeof(std::uint64_t)),
                             "cudaMemset(oversized buckets)");
    if (ok) cuda_step(cudaMemset(buffers.discarded_members, 0, sizeof(std::uint64_t)),
                             "cudaMemset(discarded members)");
    if (ok) cuda_step(detail::launch_gene_minhash(
                             buffers.support, buffers.global_rows, buffers.nonempty_genes,
                             nonempty_count, support.layout.words_per_gene, config.sketch_count,
                             config.seed, buffers.sketches, nullptr),
                         "gene MinHash launch");

    constexpr unsigned int threads = 256u;
    const unsigned int record_blocks = (unsigned int) ((record_count + threads - 1u) / threads);
    if (ok) {
        emit_lsh_records_kernel<<<record_blocks, threads>>>(
            buffers.sketches, buffers.nonempty_genes, nonempty_count, config.sketch_count,
            config.rows_per_band, config.seed, record_count, buffers.keys_a, buffers.values_a);
        cuda_step(cudaGetLastError(), "emit LSH records launch");
    }

    std::size_t sort_record_temp = 0u, scan_head_temp = 0u;
    if (ok) cuda_step(cub::DeviceRadixSort::SortPairs(
                             nullptr, sort_record_temp, buffers.keys_a, buffers.keys_b,
                             buffers.values_a, buffers.values_b, (int) record_count),
                         "CUB LSH sort scratch query");
    if (ok) cuda_step(cub::DeviceScan::ExclusiveSum(
                             nullptr, scan_head_temp, buffers.heads, buffers.bucket_ids,
                             (int) record_count), "CUB bucket scan scratch query");
    std::size_t cub_bytes = std::max(sort_record_temp, scan_head_temp);
    allocate(&buffers.cub_temp, cub_bytes, "cudaMalloc(CUB LSH scratch)");
    if (ok) cuda_step(cub::DeviceRadixSort::SortPairs(
                             buffers.cub_temp, sort_record_temp, buffers.keys_a, buffers.keys_b,
                             buffers.values_a, buffers.values_b, (int) record_count),
                         "CUB stable LSH radix sort");
    if (ok) {
        mark_bucket_heads_kernel<<<record_blocks, threads>>>(
            buffers.keys_b, buffers.values_b, record_count, buffers.heads);
        cuda_step(cudaGetLastError(), "mark LSH bucket heads launch");
    }
    if (ok) cuda_step(cub::DeviceScan::ExclusiveSum(
                             buffers.cub_temp, scan_head_temp, buffers.heads, buffers.bucket_ids,
                             (int) record_count), "CUB bucket-id exclusive scan");

    ct::u32 last_bucket_id = 0u, last_head = 0u;
    if (ok) cuda_step(cudaMemcpy(&last_bucket_id, buffers.bucket_ids + record_count - 1u,
                                 sizeof(last_bucket_id), cudaMemcpyDeviceToHost),
                             "cudaMemcpy(last bucket id D2H)");
    if (ok) cuda_step(cudaMemcpy(&last_head, buffers.heads + record_count - 1u,
                                 sizeof(last_head), cudaMemcpyDeviceToHost),
                             "cudaMemcpy(last bucket head D2H)");
    const std::uint64_t bucket_count = (std::uint64_t) last_bucket_id + last_head;
    provenance.bucket_count = bucket_count;
    if (ok) {
        scatter_bucket_offsets_kernel<<<record_blocks, threads>>>(
            buffers.heads, buffers.bucket_ids, record_count, buffers.bucket_offsets);
        cuda_step(cudaGetLastError(), "scatter bucket offsets launch");
    }

    const unsigned int bucket_blocks = (unsigned int) ((bucket_count + threads - 1u) / threads);
    if (ok) {
        count_bucket_pairs_kernel<<<bucket_blocks, threads>>>(
            buffers.bucket_offsets, bucket_count, config.maximum_bucket_size,
            buffers.pair_counts, buffers.oversized_buckets, buffers.discarded_members);
        cuda_step(cudaGetLastError(), "count bucket pairs launch");
    }
    std::size_t scan_pair_temp = 0u;
    if (ok) cuda_step(cub::DeviceScan::ExclusiveSum(
                             nullptr, scan_pair_temp, buffers.pair_counts, buffers.pair_offsets,
                             (int) bucket_count), "CUB pair-offset scan scratch query");
    if (ok && scan_pair_temp > cub_bytes) {
        cuda_step(cudaFree(buffers.cub_temp), "cudaFree(CUB LSH scratch)");
        buffers.cub_temp = nullptr;
        cub_bytes = scan_pair_temp;
        allocate(&buffers.cub_temp, cub_bytes, "cudaMalloc(CUB pair scan scratch)");
    }
    if (ok) cuda_step(cub::DeviceScan::ExclusiveSum(
                             buffers.cub_temp, scan_pair_temp, buffers.pair_counts,
                             buffers.pair_offsets, (int) bucket_count),
                         "CUB pair-offset exclusive scan");

    std::uint64_t last_pair_offset = 0u, last_pair_count = 0u;
    if (ok && bucket_count != 0u) {
        cuda_step(cudaMemcpy(&last_pair_offset, buffers.pair_offsets + bucket_count - 1u,
                             sizeof(last_pair_offset), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(last pair offset D2H)");
        cuda_step(cudaMemcpy(&last_pair_count, buffers.pair_counts + bucket_count - 1u,
                             sizeof(last_pair_count), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(last pair count D2H)");
    }
    const std::uint64_t raw_pair_count = last_pair_offset + last_pair_count;
    provenance.raw_pair_occurrences = raw_pair_count;
    if (ok && raw_pair_count > config.maximum_raw_pair_occurrences) {
        detail::set_error(error, "CUDA candidate raw pairs exceeded configured budget");
        ok = false;
    }

    std::uint64_t unique_pair_count = 0u;
    if (ok && raw_pair_count != 0u) {
        const std::size_t raw_pair_bytes = (std::size_t) raw_pair_count * sizeof(std::uint64_t);
        allocate((void **) &buffers.pairs_a, raw_pair_bytes, "cudaMalloc(raw candidate pairs A)");
        allocate((void **) &buffers.pairs_b, raw_pair_bytes, "cudaMalloc(raw candidate pairs B)");
        if (ok) {
            emit_bucket_pairs_kernel<<<(unsigned int) bucket_count, threads>>>(
                buffers.keys_b, buffers.values_b, buffers.bucket_offsets, buffers.pair_offsets,
                bucket_count, config.maximum_bucket_size, config.seed, buffers.pairs_a);
            cuda_step(cudaGetLastError(), "emit bounded candidate pairs launch");
        }
        std::size_t sort_pair_temp = 0u, unique_temp = 0u;
        if (ok) cuda_step(cub::DeviceRadixSort::SortKeys(
                                 nullptr, sort_pair_temp, buffers.pairs_a, buffers.pairs_b,
                                 (int) raw_pair_count), "CUB candidate sort scratch query");
        if (ok) cuda_step(cub::DeviceSelect::Unique(
                                 nullptr, unique_temp, buffers.pairs_b, buffers.pairs_a,
                                 buffers.unique_count, (int) raw_pair_count),
                             "CUB candidate unique scratch query");
        const std::size_t pair_cub_bytes = std::max(sort_pair_temp, unique_temp);
        if (ok && pair_cub_bytes > cub_bytes) {
            cuda_step(cudaFree(buffers.cub_temp), "cudaFree(CUB pair scan scratch)");
            buffers.cub_temp = nullptr;
            cub_bytes = pair_cub_bytes;
            allocate(&buffers.cub_temp, cub_bytes, "cudaMalloc(CUB candidate scratch)");
        }
        if (ok) cuda_step(cub::DeviceRadixSort::SortKeys(
                                 buffers.cub_temp, sort_pair_temp, buffers.pairs_a, buffers.pairs_b,
                                 (int) raw_pair_count), "CUB candidate radix sort");
        if (ok) cuda_step(cub::DeviceSelect::Unique(
                                 buffers.cub_temp, unique_temp, buffers.pairs_b, buffers.pairs_a,
                                 buffers.unique_count, (int) raw_pair_count),
                             "CUB candidate unique");
        int unique_count = 0;
        if (ok) cuda_step(cudaMemcpy(&unique_count, buffers.unique_count, sizeof(unique_count),
                                     cudaMemcpyDeviceToHost), "cudaMemcpy(unique count D2H)");
        if (ok && unique_count < 0) {
            detail::set_error(error, "CUDA candidate unique count is negative");
            ok = false;
        }
        unique_pair_count = (std::uint64_t) std::max(unique_count, 0);
    }

    std::uint64_t oversized_buckets = 0u, discarded_members = 0u;
    if (ok) cuda_step(cudaMemcpy(&oversized_buckets, buffers.oversized_buckets,
                                 sizeof(oversized_buckets), cudaMemcpyDeviceToHost),
                             "cudaMemcpy(oversized buckets D2H)");
    if (ok) cuda_step(cudaMemcpy(&discarded_members, buffers.discarded_members,
                                 sizeof(discarded_members), cudaMemcpyDeviceToHost),
                             "cudaMemcpy(discarded members D2H)");
    provenance.oversized_bucket_count = oversized_buckets;
    provenance.discarded_bucket_members = discarded_members;
    provenance.unique_candidate_count = unique_pair_count;
    provenance.device_cub_temporary_bytes = cub_bytes;

    std::size_t raw_pair_bytes = 0u;
    std::size_t device_peak_bytes = 0u;
    auto include_device_bytes = [&](std::size_t bytes) {
        if (ok && !detail::checked_add(device_peak_bytes, bytes, &device_peak_bytes)) {
            detail::set_error(error, "CUDA candidate peak allocation size overflows size_t");
            ok = false;
        }
    };
    if (!detail::checked_multiply((std::size_t) raw_pair_count,
                                  sizeof(std::uint64_t), &raw_pair_bytes)) {
        detail::set_error(error, "CUDA candidate raw-pair allocation size overflows size_t");
        ok = false;
    }
    include_device_bytes(support_bytes);
    include_device_bytes(mapping_bytes);
    include_device_bytes(nonempty_bytes);
    include_device_bytes(sketch_bytes);
    for (unsigned int i = 0u; i < 5u; ++i) include_device_bytes(record_u64_bytes);
    for (unsigned int i = 0u; i < 2u; ++i) include_device_bytes(record_u32_bytes);
    for (unsigned int i = 0u; i < 2u; ++i) include_device_bytes(record_offset_bytes);
    include_device_bytes(2u * sizeof(std::uint64_t));
    include_device_bytes(sizeof(int));
    include_device_bytes(raw_pair_bytes);
    include_device_bytes(raw_pair_bytes);
    include_device_bytes(cub_bytes);
    provenance.device_peak_bytes = device_peak_bytes;

    std::unique_ptr<gene_candidate_pair[]> host_pairs;
    if (ok && unique_pair_count != 0u) {
        host_pairs.reset(new (std::nothrow) gene_candidate_pair[(std::size_t) unique_pair_count]);
        if (host_pairs == nullptr) {
            detail::set_error(error, "failed to allocate CUDA candidate host output");
            ok = false;
        }
    }
    if (ok && unique_pair_count != 0u) {
        const unsigned int blocks = (unsigned int) ((unique_pair_count + threads - 1u) / threads);
        unpack_candidate_pairs_kernel<<<blocks, threads>>>(buffers.pairs_a, unique_pair_count);
        cuda_step(cudaGetLastError(), "unpack candidate pairs launch");
        cuda_step(cudaDeviceSynchronize(), "candidate discovery execution");
        cuda_step(cudaMemcpy(host_pairs.get(), buffers.pairs_a,
                             (std::size_t) unique_pair_count * sizeof(gene_candidate_pair),
                             cudaMemcpyDeviceToHost), "cudaMemcpy(candidate pairs D2H)");
    } else if (ok) {
        cuda_step(cudaDeviceSynchronize(), "candidate discovery execution");
    }

    const cudaError_t free_status = free_device_buffers(&buffers);
    if (ok && free_status != cudaSuccess) ok = cuda_ok(free_status, "cudaFree(candidate buffers)", error);
    if (previous_device != device) {
        const cudaError_t restore_status = cudaSetDevice(previous_device);
        if (ok && restore_status != cudaSuccess) {
            ok = cuda_ok(restore_status, "cudaSetDevice(restore)", error);
        }
    }
    if (!ok) return false;
    *out = owned_gene_candidates(std::move(host_pairs), unique_pair_count, std::move(provenance));
    return true;
}

} // namespace cellerator::compute::gene_candidates
