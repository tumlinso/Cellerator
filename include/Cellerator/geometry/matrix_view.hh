#pragma once

#include "Cellerator/geometry/format.hh"
#include "Cellerator/geometry/validate.hh"

namespace cellpack {

struct csr_view {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 nnz_count = 0u;
    const u32 *row_offsets = nullptr;
    const u32 *feature_ids = nullptr;
    const float *values = nullptr;
};

struct coo_view {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 nnz_count = 0u;
    const u32 *row_ids = nullptr;
    const u32 *feature_ids = nullptr;
    const float *values = nullptr;
};

validation_result validate_csr_view(const csr_view &view);

validation_result validate_coo_view(const coo_view &view);

} // namespace cellpack
