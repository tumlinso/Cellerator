#pragma once

#include "record_table.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cellerator::compute::graph {

struct TimeSlabSpan {
    std::uint32_t slab_id = 0;
    std::int32_t embryo_id = -1;
    std::uint32_t row_begin = 0;
    std::uint32_t row_end = 0;
    float time_begin = 0.0f;
    float time_end = 0.0f;
    bool is_delta = false;
};

struct FutureWindowBounds {
    std::vector<std::uint32_t> row_begin;
    std::vector<std::uint32_t> row_end;
};

struct DeltaSlabAssignment {
    std::uint32_t input_row = 0;
    std::uint32_t slab_id = std::numeric_limits<std::uint32_t>::max();
};

inline std::vector<TimeSlabSpan> build_time_slabs(const TrajectoryRecordTable &table, std::uint32_t target_rows_per_slab) {
    table.validate();
    if (target_rows_per_slab == 0) throw std::invalid_argument("target_rows_per_slab must be > 0");

    std::vector<TimeSlabSpan> slabs;
    std::uint32_t slab_id = 0;
    for (const EmbryoRowSpan &span : build_embryo_row_spans(table)) {
        for (std::uint32_t begin = span.row_begin; begin < span.row_end; begin += target_rows_per_slab) {
            const std::uint32_t end = std::min(begin + target_rows_per_slab, span.row_end);
            slabs.push_back(TimeSlabSpan{
                slab_id++,
                span.embryo_id,
                begin,
                end,
                table.developmental_time[begin],
                table.developmental_time[end - 1],
                false
            });
        }
    }
    return slabs;
}

inline FutureWindowBounds build_future_window_bounds(
    const TrajectoryRecordTable &table,
    float min_time_delta,
    float max_time_delta,
    float strict_future_epsilon) {
    table.validate();
    if (min_time_delta < 0.0f) throw std::invalid_argument("min_time_delta must be >= 0");
    if (max_time_delta < min_time_delta) throw std::invalid_argument("max_time_delta must be >= min_time_delta");
    if (strict_future_epsilon < 0.0f) throw std::invalid_argument("strict_future_epsilon must be >= 0");

    FutureWindowBounds bounds;
    bounds.row_begin.assign(table.rows, 0);
    bounds.row_end.assign(table.rows, 0);
    const std::vector<EmbryoRowSpan> spans = build_embryo_row_spans(table);

    for (const EmbryoRowSpan &span : spans) {
        const auto time_begin = table.developmental_time.begin() + span.row_begin;
        const auto time_end = table.developmental_time.begin() + span.row_end;
        for (std::uint32_t row = span.row_begin; row < span.row_end; ++row) {
            const float t = table.developmental_time[row];
            const float lower = t + strict_future_epsilon + min_time_delta;
            const float upper = t + max_time_delta;
            const auto lo_it = std::upper_bound(time_begin, time_end, lower);
            const auto hi_it = std::upper_bound(lo_it, time_end, upper);
            bounds.row_begin[row] = static_cast<std::uint32_t>(lo_it - table.developmental_time.begin());
            bounds.row_end[row] = static_cast<std::uint32_t>(hi_it - table.developmental_time.begin());
        }
    }
    return bounds;
}

inline std::vector<DeltaSlabAssignment> assign_rows_to_delta_slabs(
    const std::vector<TimeSlabSpan> &base_slabs,
    const TrajectoryRecordTable &new_rows,
    float halo_time) {
    new_rows.validate();
    if (halo_time < 0.0f) throw std::invalid_argument("halo_time must be >= 0");

    std::vector<DeltaSlabAssignment> assignments(new_rows.rows);
    for (std::uint32_t row = 0; row < new_rows.rows; ++row) {
        assignments[row].input_row = row;
        assignments[row].slab_id = std::numeric_limits<std::uint32_t>::max();
        const std::int32_t embryo = new_rows.embryo_id[row];
        const float time = new_rows.developmental_time[row];

        for (const TimeSlabSpan &slab : base_slabs) {
            if (slab.embryo_id != embryo) continue;
            if (time >= slab.time_begin - halo_time && time <= slab.time_end + halo_time) {
                assignments[row].slab_id = slab.slab_id;
                break;
            }
        }
    }
    return assignments;
}

} // namespace cellerator::compute::graph
