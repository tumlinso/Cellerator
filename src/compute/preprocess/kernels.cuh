#pragma once

#include "types.cuh"
#include "../primitives/warp_reduce.cuh"

#include <cuda_runtime.h>

namespace cellerator {
namespace compute {
namespace preprocess {
namespace kernels {

namespace reduce = ::cellerator::compute::primitives::reduce;

#include "kernels/compute_cell_metrics_kernel.cuh"
#include "kernels/compute_cell_metrics_blocked_ell_kernel.cuh"
#include "kernels/compute_cell_metrics_sliced_ell_kernel.cuh"
#include "kernels/normalize_log1p_kernel.cuh"
#include "kernels/normalize_log1p_blocked_ell_kernel.cuh"
#include "kernels/normalize_log1p_sliced_ell_kernel.cuh"
#include "kernels/square_values_kernel.cuh"
#include "kernels/convert_values_kernel.cuh"
#include "kernels/fill_ones_kernel.cuh"
#include "kernels/add_scalar_kernel.cuh"
#include "kernels/expand_keep_mask_kernel.cuh"
#include "kernels/build_gene_filter_mask_kernel.cuh"
#include "kernels/accumulate_gene_metrics_blocked_ell_kernel.cuh"
#include "kernels/accumulate_gene_metrics_sliced_ell_kernel.cuh"

} // namespace kernels
} // namespace preprocess
} // namespace compute
} // namespace cellerator
