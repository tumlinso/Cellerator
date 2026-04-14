#pragma once

#include "types.cuh"
#include "rebuild_policy.cuh"
#include "../../../extern/CellShard/src/convert/filtered_blocked_ell_to_compressed.cuh"

namespace cellerator {
namespace compute {
namespace preprocess {

namespace csc = ::cellshard::convert;

struct alignas(16) filtered_execution_view_result {
    csc::filtered_blocked_ell_result filtered;
    rebuild_decision decision;
};

struct alignas(16) filtered_execution_view_workspace {
    csc::filtered_blocked_ell_workspace convert;
};

static inline void init(filtered_execution_view_workspace *ws) {
    csc::init(&ws->convert);
}

static inline void clear(filtered_execution_view_workspace *ws) {
    csc::clear(&ws->convert);
}

static inline int setup(filtered_execution_view_workspace *ws,
                        int device,
                        cudaStream_t stream = (cudaStream_t) 0) {
    return csc::setup(&ws->convert, device, stream);
}

static inline int build_filtered_execution_view(
    const csv::blocked_ell_view *src,
    unsigned int output_rows,
    unsigned int output_cols,
    const unsigned char *d_keep_rows,
    const unsigned char *d_keep_cols,
    const unsigned int *d_row_remap,
    const unsigned int *d_col_remap,
    unsigned int requested_bucket_count,
    const rebuild_policy &policy,
    filtered_execution_view_workspace *ws,
    filtered_execution_view_result *out
) {
    filtered_execution_view_result local{};
    std::size_t rebuild_bytes = 0u;

    if (src == 0 || ws == 0) return 0;
    if (!csc::build_bucketed_filtered_blocked_ell_major_view(src,
                                                             output_rows,
                                                             output_cols,
                                                             d_keep_rows,
                                                             d_keep_cols,
                                                             d_row_remap,
                                                             d_col_remap,
                                                             requested_bucket_count,
                                                             &ws->convert,
                                                             &local.filtered)) return 0;
    rebuild_bytes = estimate_filtered_compressed_bytes(local.filtered.stats);
    local.decision = evaluate_rebuild_policy(local.filtered.stats, rebuild_bytes, policy);
    if (out != 0) *out = local;
    return 1;
}

} // namespace preprocess
} // namespace compute
} // namespace cellerator
