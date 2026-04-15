#pragma once

#include "portable_backend.cuh"
#include "extreme_backend.cuh"
#include "cellerator_cuda_mode.hh"

namespace cellerator::quantized {

struct v100_launch_policy {
    enum {
        threads = build::cuda_mode_is_native_extreme
            ? static_cast<int>(extreme_backend::launch_policy::threads)
            : static_cast<int>(portable_backend::launch_policy::threads),
        min_blocks_per_sm = build::cuda_mode_is_native_extreme
            ? static_cast<int>(extreme_backend::launch_policy::min_blocks_per_sm)
            : static_cast<int>(portable_backend::launch_policy::min_blocks_per_sm)
    };
};

} // namespace cellerator::quantized
