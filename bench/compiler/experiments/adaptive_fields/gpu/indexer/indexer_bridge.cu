#include "indexer_bridge.h"

#include <Cellerator/compute/indexing/cuda_postings.cuh>
#include <Cellerator/compute/support/logical_support.hh>

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ix = cellerator::compute::indexing;
namespace support = cellerator::compute::support;

namespace {
thread_local std::string last_error;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <class T>
struct device_buffer {
  T* pointer = nullptr;
  std::size_t count = 0;
  device_buffer() = default;
  explicit device_buffer(std::size_t n) : count(n) {
    if (count) check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer),
                                     count * sizeof(T)), "cudaMalloc");
  }
  device_buffer(const device_buffer&) = delete;
  device_buffer& operator=(const device_buffer&) = delete;
  device_buffer(device_buffer&& other) noexcept
      : pointer(other.pointer), count(other.count) {
    other.pointer = nullptr; other.count = 0;
  }
  device_buffer& operator=(device_buffer&& other) noexcept {
    if (this != &other) {
      if (pointer) cudaFree(pointer);
      pointer = other.pointer; count = other.count;
      other.pointer = nullptr; other.count = 0;
    }
    return *this;
  }
  ~device_buffer() { if (pointer) cudaFree(pointer); }
};

template <class T>
void upload(device_buffer<T>& target, const std::vector<T>& source,
            cudaStream_t stream) {
  if (!source.empty())
    check_cuda(cudaMemcpyAsync(target.pointer, source.data(),
        source.size() * sizeof(T), cudaMemcpyHostToDevice, stream), "H2D");
}

__global__ void scan_flags_kernel(
    const uint64_t* sources, const uint64_t* destinations, uint64_t edges,
    uint64_t cells, const uint64_t* query_offsets,
    const ix::posting_term_v1* queries, uint64_t* flags) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = edges * cells;
  if (index >= total) return;
  const uint64_t cell = index / edges;
  const uint64_t ordinal = index - cell * edges;
  bool matched = false;
  for (uint64_t query = query_offsets[cell]; query < query_offsets[cell + 1]; ++query) {
    const auto term = queries[query];
    if ((term.role == 0 && term.entity_id == sources[ordinal] + 1) ||
        (term.role == 1 && term.entity_id == destinations[ordinal] + 1)) {
      matched = true;
      break;
    }
  }
  flags[index] = matched ? 1u : 0u;
}

__global__ void scan_row_offsets_kernel(
    const uint64_t* flags, const uint64_t* prefix, uint64_t edges,
    uint64_t cells, uint64_t* row_offsets) {
  if (blockIdx.x || threadIdx.x) return;
  row_offsets[0] = 0;
  for (uint64_t cell = 0; cell < cells; ++cell) {
    const uint64_t last = (cell + 1) * edges - 1;
    row_offsets[cell + 1] = prefix[last] + flags[last];
  }
}

__global__ void scan_emit_kernel(
    const uint64_t* flags, const uint64_t* prefix, const uint64_t* edge_ids,
    uint64_t edges, uint64_t cells, ix::indexed_work_item_v1* items) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= edges * cells || flags[index] == 0) return;
  const uint64_t ordinal = index % edges;
  items[prefix[index]] = {edge_ids[ordinal], ordinal};
}

bool ordered_row(const ix::reverse_posting_view_v1& index,
                 const std::vector<AFPostingTermV1>& queries,
                 uint64_t begin, uint64_t end) {
  if (begin > end) return false;
  bool have_previous_key = false;
  uint32_t previous_role = 0;
  uint64_t previous_entity = 0;
  bool have_previous_id = false;
  uint64_t previous_id = 0;
  for (uint64_t qi = begin; qi < end; ++qi) {
    const auto& query = queries[qi];
    if (query.reserved) return false;
    if (have_previous_key &&
        !(previous_role < query.role ||
          (previous_role == query.role && previous_entity < query.entity_id)))
      return false;
    previous_role = query.role;
    previous_entity = query.entity_id;
    have_previous_key = true;
    auto first = std::lower_bound(index.entries, index.entries + index.entry_count,
        ix::reverse_posting_record_v1{query.role, 0u, query.entity_id, 0u, 0u},
        [](const auto& a, const auto& b) {
          return a.role < b.role || (a.role == b.role && a.entity_id < b.entity_id);
        });
    for (auto it = first; it != index.entries + index.entry_count &&
         it->role == query.role && it->entity_id == query.entity_id; ++it) {
      if (have_previous_id && it->logical_edge_id <= previous_id) return false;
      previous_id = it->logical_edge_id;
      have_previous_id = true;
    }
  }
  return true;
}

} // namespace

struct AFIndexHandleV1 {
  uint64_t entity_count = 0;
  uint64_t epoch = 0;
  uint64_t relation_id = 0;
  uint64_t structure_id = 0;
  bool postings_ready = false;
  std::vector<uint64_t> edge_ids, sources, destinations;
  std::vector<support::logical_support_edge_v1> support_edges;
  std::vector<support::logical_support_participant_v1> participants;
  support::logical_support_view_v1 support_view{};
  support::support_snapshot_pair_v1 snapshot{};
  std::vector<ix::reverse_posting_record_v1> record_scratch, host_postings;
  ix::reverse_posting_view_v1 host_index{};
  std::vector<ix::posting_term_v1> projection_terms;
  std::vector<uint64_t> projection_offsets;
  std::vector<ix::indexed_work_item_v1> projection_items;
  ix::cuda_posting_prepare_requirements_v1 projection_requirements{};
  device_buffer<uint64_t> d_edge_ids, d_sources, d_destinations;
  device_buffer<ix::posting_term_v1> d_terms;
  device_buffer<uint64_t> d_term_offsets;
  device_buffer<ix::indexed_work_item_v1> d_posting_items;
  ix::cuda_reverse_posting_view_v1 device_index{};
  cudaStream_t stream = nullptr;

  ~AFIndexHandleV1() { if (stream) cudaStreamDestroy(stream); }
};

namespace {
void host_scan_oracle(const AFIndexHandleV1* handle, uint64_t cells,
                      const std::vector<uint64_t>& query_offsets,
                      const std::vector<AFPostingTermV1>& queries,
                      std::vector<uint64_t>& offsets,
                      std::vector<uint64_t>& ids,
                      std::vector<uint64_t>& ordinals) {
  offsets.assign(static_cast<std::size_t>(cells + 1), 0);
  for (uint64_t cell = 0; cell < cells; ++cell) {
    std::vector<ix::indexed_work_item_v1> row;
    for (uint64_t ordinal = 0; ordinal < handle->edge_ids.size(); ++ordinal) {
      bool match = false;
      for (uint64_t q = query_offsets[cell]; q < query_offsets[cell + 1]; ++q) {
        const auto term = queries[q];
        if ((term.role == 0 && term.entity_id == handle->sources[ordinal] + 1) ||
            (term.role == 1 && term.entity_id == handle->destinations[ordinal] + 1)) {
          match = true; break;
        }
      }
      if (match) row.push_back({handle->edge_ids[ordinal], ordinal});
    }
    for (const auto& item : row) {
      ids.push_back(item.logical_edge_id);
      ordinals.push_back(item.logical_edge_ordinal);
    }
    offsets[cell + 1] = ids.size();
  }
}
} // namespace

extern "C" AFIndexHandleV1* af_index_create_v1(
    uint64_t edge_count, const uint64_t* edge_ids,
    const uint64_t* sources, const uint64_t* destinations,
    uint64_t entity_count, uint64_t structure_epoch,
    uint64_t relation_id, uint64_t structure_id) {
  try {
    last_error.clear();
    if (!edge_count || !edge_ids || !sources || !destinations ||
        !entity_count || !structure_epoch || !relation_id || !structure_id)
      throw std::invalid_argument("support dimensions and identities must be nonzero");
    auto handle = std::make_unique<AFIndexHandleV1>();
    handle->entity_count = entity_count;
    handle->epoch = structure_epoch;
    handle->relation_id = relation_id;
    handle->structure_id = structure_id;
    handle->edge_ids.assign(edge_ids, edge_ids + edge_count);
    handle->sources.assign(sources, sources + edge_count);
    handle->destinations.assign(destinations, destinations + edge_count);
    handle->support_edges.resize(edge_count);
    handle->participants.resize(edge_count * 2);
    for (uint64_t i = 0; i < edge_count; ++i) {
      if (!handle->edge_ids[i] || (i && handle->edge_ids[i - 1] >= handle->edge_ids[i]) ||
          handle->sources[i] >= entity_count || handle->destinations[i] >= entity_count)
        throw std::invalid_argument("edge IDs must be sorted unique and endpoints in range");
      handle->support_edges[i] = {handle->edge_ids[i], 2 * i, 2, 0};
      handle->participants[2 * i] = {0, 0, handle->sources[i] + 1};
      handle->participants[2 * i + 1] = {1, 0, handle->destinations[i] + 1};
    }
    handle->support_view = {support::logical_support_schema_version_v1, 0,
        relation_id, structure_id, structure_epoch,
        handle->support_edges.data(), handle->support_edges.size(),
        handle->participants.data(), handle->participants.size()};
    handle->snapshot = {&handle->support_view, nullptr, structure_epoch};
    if (support::validate_support_snapshot_pair_v1(handle->snapshot) !=
        support::support_snapshot_status_v1::success)
      throw std::invalid_argument("support snapshot validation failed");

    check_cuda(cudaStreamCreateWithFlags(&handle->stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
    handle->d_edge_ids = device_buffer<uint64_t>(edge_count);
    handle->d_sources = device_buffer<uint64_t>(edge_count);
    handle->d_destinations = device_buffer<uint64_t>(edge_count);
    upload(handle->d_edge_ids, handle->edge_ids, handle->stream);
    upload(handle->d_sources, handle->sources, handle->stream);
    upload(handle->d_destinations, handle->destinations, handle->stream);
    check_cuda(cudaStreamSynchronize(handle->stream), "prepare support sync");
    return handle.release();
  } catch (const std::exception& error) {
    last_error = error.what();
    return nullptr;
  }
}

extern "C" uint32_t af_index_prepare_ordered_postings_v1(
    AFIndexHandleV1* handle) {
  try {
    last_error.clear();
    if (!handle) return AF_INDEX_INVALID_ARGUMENT_V1;
    if (handle->postings_ready) return AF_INDEX_SUCCESS_V1;
    const auto required = ix::query_reverse_posting_requirements_v1(handle->snapshot);
    handle->record_scratch.resize(required.record_capacity);
    handle->host_postings.resize(required.maximum_unique_entries);
    if (ix::build_reverse_postings_v1(handle->snapshot,
            handle->record_scratch.data(), handle->record_scratch.size(),
            handle->host_postings.data(), handle->host_postings.size(),
            &handle->host_index) != ix::posting_status_v1::success)
      throw std::runtime_error("canonical host reverse-posting build failed");
    handle->projection_requirements = ix::cuda_posting_projection_requirements_v1(
        handle->host_index);
    const auto& prep = handle->projection_requirements;
    handle->projection_terms.resize(prep.term_count);
    handle->projection_offsets.resize(prep.term_count + 1);
    handle->projection_items.resize(prep.item_count);
    ix::cuda_posting_prepare_requirements_v1 exact{};
    if (ix::build_cuda_posting_projection_v1(handle->host_index,
            handle->projection_terms.data(), handle->projection_terms.size(),
            handle->projection_offsets.data(), handle->projection_offsets.size(),
            handle->projection_items.data(), handle->projection_items.size(),
            &exact) != ix::cuda_posting_prepare_status_v1::success ||
        exact.term_count != prep.term_count || exact.item_count != prep.item_count)
      throw std::runtime_error("canonical CUDA posting projection build failed");
    handle->d_terms = device_buffer<ix::posting_term_v1>(prep.term_count);
    handle->d_term_offsets = device_buffer<uint64_t>(prep.term_count + 1);
    handle->d_posting_items = device_buffer<ix::indexed_work_item_v1>(prep.item_count);
    upload(handle->d_terms, handle->projection_terms, handle->stream);
    upload(handle->d_term_offsets, handle->projection_offsets, handle->stream);
    upload(handle->d_posting_items, handle->projection_items, handle->stream);
    check_cuda(cudaStreamSynchronize(handle->stream), "prepare projection sync");
    handle->device_index = {handle->relation_id, handle->structure_id, handle->epoch,
        handle->d_terms.pointer, handle->d_term_offsets.pointer, prep.term_count,
        handle->d_posting_items.pointer, prep.item_count};
    handle->postings_ready = true;
    return AF_INDEX_SUCCESS_V1;
  } catch (const std::exception& error) {
    last_error = error.what();
    return AF_INDEX_POSTING_ERROR_V1;
  }
}

extern "C" uint32_t af_index_capacity_v1(const AFIndexHandleV1* handle,
                                          uint64_t cells, uint64_t* capacity) {
  if (!handle || !cells || !capacity) return AF_INDEX_INVALID_ARGUMENT_V1;
  if (handle->edge_ids.size() > std::numeric_limits<uint64_t>::max() / cells) {
    last_error = "E*cells capacity overflows uint64";
    return AF_INDEX_INVALID_ARGUMENT_V1;
  }
  *capacity = static_cast<uint64_t>(handle->edge_ids.size()) * cells;
  return AF_INDEX_SUCCESS_V1;
}

extern "C" uint32_t af_index_run_v1(
    AFIndexHandleV1* handle, uint32_t mode,
    uint64_t cells, const uint64_t* query_offsets,
    const AFPostingTermV1* queries, uint64_t query_count,
    uint64_t state_generation, int verify, AFIndexOutputV1* output) {
  try {
    last_error.clear();
    if (!handle || !cells || !query_offsets || (query_count && !queries) || !output ||
        mode > AF_INDEX_ORDERED_POSTINGS_V1 ||
        !output->row_offsets || output->row_offsets_capacity < cells + 1 ||
        (output->item_capacity && (!output->logical_edge_ids || !output->edge_ordinals ||
                                   !output->gate_value_indices)))
      return AF_INDEX_INVALID_ARGUMENT_V1;
    const uint64_t edges = handle->edge_ids.size();
    if (edges > std::numeric_limits<uint64_t>::max() / cells)
      return AF_INDEX_INVALID_ARGUMENT_V1;
    const auto max_items = edges * cells;
    if (output->item_capacity < max_items) {
      output->required_capacity = max_items;
      return AF_INDEX_INSUFFICIENT_CAPACITY_V1;
    }
    if (mode == AF_INDEX_ORDERED_POSTINGS_V1 && !handle->postings_ready) {
      last_error = "ordered postings mode has not been prepared";
      return AF_INDEX_MODE_NOT_PREPARED_V1;
    }
    std::vector<uint64_t> host_offsets(query_offsets, query_offsets + cells + 1);
    std::vector<AFPostingTermV1> host_queries;
    if (query_count) host_queries.assign(queries, queries + query_count);
    if (host_offsets.front() != 0 || host_offsets.back() != query_count)
      return AF_INDEX_INVALID_ARGUMENT_V1;
    for (uint64_t i = 0; i < cells; ++i)
      if (host_offsets[i] > host_offsets[i + 1] || host_offsets[i + 1] > query_count)
        return AF_INDEX_INVALID_ARGUMENT_V1;

    std::vector<uint64_t> host_expected_offsets;
    std::vector<uint64_t> host_expected_ids, host_expected_ordinals;
    if (verify)
      host_scan_oracle(handle, cells, host_offsets, host_queries,
                       host_expected_offsets, host_expected_ids,
                       host_expected_ordinals);
    uint64_t fast_rows = 0, fallback_rows = 0;
    if (mode == AF_INDEX_ORDERED_POSTINGS_V1 && verify) {
      for (uint64_t cell = 0; cell < cells; ++cell) {
        if (ordered_row(handle->host_index, host_queries, host_offsets[cell],
                        host_offsets[cell + 1])) ++fast_rows;
        else ++fallback_rows;
      }
    }

    device_buffer<uint64_t> d_query_offsets(cells + 1);
    device_buffer<ix::posting_term_v1> d_queries(std::max<uint64_t>(query_count, 1));
    device_buffer<uint64_t> d_output_offsets(cells + 1);
    device_buffer<ix::indexed_work_item_v1> d_items(std::max<uint64_t>(max_items, 1));
    upload(d_query_offsets, host_offsets, handle->stream);
    std::vector<ix::posting_term_v1> native_queries;
    native_queries.reserve(host_queries.size());
    for (const auto& query : host_queries)
      native_queries.push_back({query.role, query.reserved, query.entity_id});
    upload(d_queries, native_queries, handle->stream);

    if (mode == AF_INDEX_SCAN_V1) {
      device_buffer<uint64_t> d_flags(max_items), d_prefix(max_items);
      size_t scan_bytes = 0;
      check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes,
          d_flags.pointer, d_prefix.pointer, max_items, handle->stream), "CUB scan sizing");
      device_buffer<unsigned char> d_scan_temp(scan_bytes);
      constexpr uint32_t threads = 256;
      const uint64_t blocks = (max_items + threads - 1) / threads;
      scan_flags_kernel<<<static_cast<uint32_t>(blocks), threads, 0, handle->stream>>>(
          handle->d_sources.pointer, handle->d_destinations.pointer, edges,
          cells, d_query_offsets.pointer, d_queries.pointer, d_flags.pointer);
      check_cuda(cudaGetLastError(), "scan flags launch");
      check_cuda(cub::DeviceScan::ExclusiveSum(d_scan_temp.pointer, scan_bytes,
          d_flags.pointer, d_prefix.pointer, max_items, handle->stream), "CUB exclusive scan");
      scan_row_offsets_kernel<<<1, 1, 0, handle->stream>>>(
          d_flags.pointer, d_prefix.pointer, edges, cells, d_output_offsets.pointer);
      check_cuda(cudaGetLastError(), "scan row offsets launch");
      scan_emit_kernel<<<static_cast<uint32_t>(blocks), threads, 0, handle->stream>>>(
          d_flags.pointer, d_prefix.pointer, handle->d_edge_ids.pointer,
          edges, cells, d_items.pointer);
      check_cuda(cudaGetLastError(), "scan emit launch");
    } else {
      device_buffer<ix::cuda_posting_scratch_v1::cursor_v1> d_cursors(
          std::max<uint64_t>(query_count, 1));
      device_buffer<uint64_t> d_counts(cells);
      device_buffer<ix::cuda_posting_status_v1> d_status(1);
      ix::cuda_posting_scratch_v1 scratch{d_cursors.pointer,
          std::max<uint64_t>(query_count, 1), d_counts.pointer, d_status.pointer};
      ix::cuda_posting_output_v1 result{d_output_offsets.pointer, cells + 1,
          d_items.pointer, max_items};
      check_cuda(ix::cuda_query_reverse_postings_any_v1(handle->device_index,
          handle->epoch, cells, d_query_offsets.pointer, d_queries.pointer,
          query_count, result, scratch, handle->stream), "posting query launch");
      ix::cuda_posting_status_v1 status{};
      check_cuda(cudaMemcpyAsync(&status, d_status.pointer, sizeof(status),
          cudaMemcpyDeviceToHost, handle->stream), "posting status D2H");
      check_cuda(cudaStreamSynchronize(handle->stream), "posting sync");
      if (status.code != 0) {
        last_error = "native posting status=" + std::to_string(status.code) +
                     " required=" + std::to_string(status.required);
        return AF_INDEX_POSTING_ERROR_V1;
      }
    }
    check_cuda(cudaStreamSynchronize(handle->stream), "workset sync");
    std::vector<uint64_t> result_offsets(static_cast<std::size_t>(cells + 1));
    check_cuda(cudaMemcpy(result_offsets.data(), d_output_offsets.pointer,
        result_offsets.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost), "offset D2H");
    const uint64_t item_count = result_offsets.back();
    if (item_count > output->item_capacity || item_count > max_items) {
      output->required_capacity = item_count;
      return AF_INDEX_INSUFFICIENT_CAPACITY_V1;
    }
    std::vector<ix::indexed_work_item_v1> items(static_cast<std::size_t>(item_count));
    if (item_count)
      check_cuda(cudaMemcpy(items.data(), d_items.pointer,
          items.size() * sizeof(ix::indexed_work_item_v1), cudaMemcpyDeviceToHost),
          "workset items D2H");
    for (uint64_t i = 0; i < item_count; ++i) {
      output->logical_edge_ids[i] = items[i].logical_edge_id;
      output->edge_ordinals[i] = items[i].logical_edge_ordinal;
      output->gate_value_indices[i] = i;
    }
    std::copy(result_offsets.begin(), result_offsets.end(), output->row_offsets);
    output->item_count = item_count;
    output->structure_epoch = handle->epoch;
    output->state_generation = state_generation;
    output->ordered_fast_path_rows = fast_rows;
    output->exact_fallback_rows = fallback_rows;
    output->path_diagnostics_valid = verify ? 1u : 0u;
    output->required_capacity = item_count;
    if (verify && (result_offsets != host_expected_offsets ||
        item_count != host_expected_ids.size())) {
      last_error = "device workset CSR offsets/count differ from host scan oracle";
      return AF_INDEX_ORACLE_MISMATCH_V1;
    }
    if (verify)
      for (uint64_t i = 0; i < item_count; ++i)
        if (output->logical_edge_ids[i] != host_expected_ids[i] ||
            output->edge_ordinals[i] != host_expected_ordinals[i]) {
          last_error = "device stable identities/ordinals differ from host scan oracle";
          return AF_INDEX_ORACLE_MISMATCH_V1;
        }
    return AF_INDEX_SUCCESS_V1;
  } catch (const std::exception& error) {
    last_error = error.what();
    return AF_INDEX_CUDA_ERROR_V1;
  }
}

extern "C" void af_index_destroy_v1(AFIndexHandleV1* handle) { delete handle; }
extern "C" const char* af_index_last_error_v1(void) { return last_error.c_str(); }
