#pragma once

#include <Cellerator/layout/conversion/bucket.cuh>

namespace cellshard {
namespace bucket {

// Compatibility surface: generic major-axis bucket math is implemented by
// CelleratorCompute. CellShard keeps storage/shard wrappers.
using ::cellerator::layout::conversion::bucket::build_major_nnz_bucket_plan_raw;
using ::cellerator::layout::conversion::bucket::build_major_nnz_bucket_plan_custom_raw;
using ::cellerator::layout::conversion::bucket::build_major_nnz_bucket_plan_library_raw;
using ::cellerator::layout::conversion::bucket::build_shard_major_nnz_bucket_plan_raw;
using ::cellerator::layout::conversion::bucket::clamp_bucket_count;
using ::cellerator::layout::conversion::bucket::major_nnz_bucket_plan_view;
using ::cellerator::layout::conversion::bucket::major_nnz_bucket_scan_scratch_bytes;
using ::cellerator::layout::conversion::bucket::major_nnz_bucket_sort_scratch_bytes;
using ::cellerator::layout::conversion::bucket::rebuild_bucketed_shard_compressed_raw;
using ::cellerator::layout::conversion::bucket::rebuild_compressed_major_order_raw;

} // namespace bucket
} // namespace cellshard
