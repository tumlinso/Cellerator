#pragma once

#include "Cellerator/geometry/matrix_view.hh"
#include "Cellerator/geometry/planner.hh"

#include <vector>

namespace cellpack {

struct packed_coordinate {
    u32 original_row = 0u;
    u32 original_feature = 0u;
    u32 permuted_row = 0u;
    u32 permuted_feature = 0u;
    u32 region_id = invalid_id;
    float value = 0.0f;
};

struct packed_coordinate_plan {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    std::vector<packed_coordinate> coordinates;
};

struct reconstructed_csr {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    std::vector<u32> row_offsets;
    std::vector<u32> feature_ids;
    std::vector<float> values;
};

validation_result build_packed_coordinate_plan(
    const csr_view &source,
    const static_plan &plan,
    packed_coordinate_plan *out);

validation_result build_packed_coordinate_plan(
    const coo_view &source,
    const static_plan &plan,
    packed_coordinate_plan *out);

validation_result reconstruct_csr_from_coordinate_plan(
    u32 row_count,
    u32 feature_count,
    const static_plan &plan,
    const packed_coordinate_plan &packed,
    reconstructed_csr *out);

} // namespace cellpack
