#pragma once

#include "slab_index.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cellerator::compute::graph {

struct DeltaSlabBuffer {
    std::uint32_t slab_id = std::numeric_limits<std::uint32_t>::max();
    std::int32_t embryo_id = -1;
    float halo_time = 0.0f;
    std::vector<std::uint32_t> input_rows;
};

struct IncrementalInsertPlan {
    std::vector<DeltaSlabBuffer> slabs;
    std::vector<std::uint32_t> unassigned_rows;
};

inline IncrementalInsertPlan plan_incremental_insert(
    const std::vector<TimeSlabSpan> &base_slabs,
    const TrajectoryRecordTable &new_rows,
    float halo_time) {
    const std::vector<DeltaSlabAssignment> assignments = assign_rows_to_delta_slabs(base_slabs, new_rows, halo_time);
    IncrementalInsertPlan plan;

    for (const DeltaSlabAssignment &assignment : assignments) {
        if (assignment.slab_id == std::numeric_limits<std::uint32_t>::max()) {
            plan.unassigned_rows.push_back(assignment.input_row);
            continue;
        }

        auto it = std::find_if(plan.slabs.begin(), plan.slabs.end(), [&](const DeltaSlabBuffer &buffer) {
            return buffer.slab_id == assignment.slab_id;
        });
        if (it == plan.slabs.end()) {
            const auto slab_it = std::find_if(base_slabs.begin(), base_slabs.end(), [&](const TimeSlabSpan &slab) {
                return slab.slab_id == assignment.slab_id;
            });
            if (slab_it == base_slabs.end()) throw std::runtime_error("assigned slab_id not found");
            plan.slabs.push_back(DeltaSlabBuffer{
                assignment.slab_id,
                slab_it->embryo_id,
                halo_time,
                { assignment.input_row }
            });
        } else {
            it->input_rows.push_back(assignment.input_row);
        }
    }

    return plan;
}

} // namespace cellerator::compute::graph
