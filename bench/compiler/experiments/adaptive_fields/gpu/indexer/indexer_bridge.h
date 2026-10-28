#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AFIndexHandleV1 AFIndexHandleV1;

typedef struct AFPostingTermV1 {
  uint32_t role;
  uint32_t reserved;
  uint64_t entity_id; /* one-based logical entity ID, matching Cellerator support */
} AFPostingTermV1;

enum AFIndexModeV1 {
  AF_INDEX_SCAN_V1 = 0,
  AF_INDEX_ORDERED_POSTINGS_V1 = 1
};

enum AFIndexStatusV1 {
  AF_INDEX_SUCCESS_V1 = 0,
  AF_INDEX_INVALID_ARGUMENT_V1 = 1,
  AF_INDEX_INSUFFICIENT_CAPACITY_V1 = 2,
  AF_INDEX_CUDA_ERROR_V1 = 3,
  AF_INDEX_SUPPORT_ERROR_V1 = 4,
  AF_INDEX_POSTING_ERROR_V1 = 5,
  AF_INDEX_ORACLE_MISMATCH_V1 = 6,
  AF_INDEX_MODE_NOT_PREPARED_V1 = 7
};

typedef struct AFIndexOutputV1 {
  uint64_t* row_offsets;       /* cells + 1 */
  uint64_t row_offsets_capacity;
  uint64_t* logical_edge_ids;  /* stable IDs, sorted/unique within each cell */
  uint64_t* edge_ordinals;     /* canonical support ordinals */
  uint64_t* gate_value_indices;/* flattened [0, item_count), PerCellWorkset-ready */
  uint64_t item_capacity;
  uint64_t item_count;
  uint64_t structure_epoch;
  uint64_t state_generation;
  uint64_t ordered_fast_path_rows;
  uint64_t exact_fallback_rows;
  uint32_t path_diagnostics_valid;
  uint32_t reserved;
  uint64_t required_capacity;
} AFIndexOutputV1;

/* Prepare an immutable pair-support view and the existing Cellerator reverse
 * posting CUDA projection. Endpoint IDs are zero-based at this C ABI and are
 * converted to the native support API's one-based logical entity IDs.
 * edge_ids must be strictly increasing, unique, and nonzero. */
AFIndexHandleV1* af_index_create_v1(
    uint64_t edge_count, const uint64_t* edge_ids,
    const uint64_t* sources, const uint64_t* destinations,
    uint64_t entity_count, uint64_t structure_epoch,
    uint64_t relation_id, uint64_t structure_id);

/* Prepare the optional reverse-posting host/device projection. A scan-only
 * plan need not pay this cost. Repeated calls are idempotent. */
uint32_t af_index_prepare_ordered_postings_v1(AFIndexHandleV1* handle);

/* Return the maximum item capacity E*cells. Use checked uint64 arithmetic. */
uint32_t af_index_capacity_v1(const AFIndexHandleV1* handle,
                              uint64_t cells, uint64_t* capacity);

/* Discover and materialize an exact CSR workset. query_offsets has cells+1
 * entries, query terms are (role, one-based entity ID), and state_generation
 * is copied into the output alongside the immutable structure epoch. The
 * function synchronizes before returning host arrays so callers can wrap them
 * into PerCellWorkset tensors; this host/device transfer is part of the
 * experiment path and must be included in complete cost. Setting verify=1
 * additionally compares all output entries to an independent host scan oracle. */
uint32_t af_index_run_v1(
    AFIndexHandleV1* handle, uint32_t mode,
    uint64_t cells, const uint64_t* query_offsets,
    const AFPostingTermV1* queries, uint64_t query_count,
    uint64_t state_generation, int verify,
    AFIndexOutputV1* output);

void af_index_destroy_v1(AFIndexHandleV1* handle);
const char* af_index_last_error_v1(void);

#ifdef __cplusplus
}
#endif
