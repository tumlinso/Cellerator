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
    std::uint32_t input_begin = 0u;
    std::uint32_t input_count = 0u;
};

struct IncrementalInsertPlan {
    std::vector<DeltaSlabBuffer> slabs;
    std::vector<std::uint32_t> input_rows;
    std::vector<std::uint32_t> unassigned_rows;
};

inline IncrementalInsertPlan plan_incremental_insert(
    const std::vector<TimeSlabSpan> &base_slabs,
    const TrajectoryRecordTable &new_rows,
    float halo_time) {
    const auto assignments = assign_rows_to_delta_slabs(base_slabs, new_rows, halo_time);
    IncrementalInsertPlan plan;
    plan.slabs.reserve(base_slabs.size());
    plan.unassigned_rows.reserve(assignments.size());
    std::vector<std::uint32_t> base_to_output(
        base_slabs.size(), std::numeric_limits<std::uint32_t>::max());
    for (const DeltaSlabAssignment &assignment : assignments) {
        if (assignment.slab_id == std::numeric_limits<std::uint32_t>::max()) {
            plan.unassigned_rows.push_back(assignment.input_row);
            continue;
        }
        const auto slab_it = std::find_if(base_slabs.begin(), base_slabs.end(), [&](const TimeSlabSpan &slab) {
            return slab.slab_id == assignment.slab_id;
        });
        if (slab_it == base_slabs.end()) throw std::runtime_error("assigned slab_id not found");
        const std::size_t base_index = static_cast<std::size_t>(slab_it - base_slabs.begin());
        std::uint32_t output = base_to_output[base_index];
        if (output == std::numeric_limits<std::uint32_t>::max()) {
            output = static_cast<std::uint32_t>(plan.slabs.size());
            base_to_output[base_index] = output;
            plan.slabs.push_back(DeltaSlabBuffer{
                assignment.slab_id,
                slab_it->embryo_id,
                halo_time,
                0u,
                0u
            });
        }
        ++plan.slabs[output].input_count;
    }
    std::uint32_t total = 0u;
    for (DeltaSlabBuffer &slab : plan.slabs) {
        slab.input_begin = total;
        total += slab.input_count;
    }
    plan.input_rows.resize(total);
    for (DeltaSlabBuffer &slab : plan.slabs) slab.input_count = slab.input_begin;
    for (const DeltaSlabAssignment &assignment : assignments) {
        if (assignment.slab_id == std::numeric_limits<std::uint32_t>::max()) continue;
        const auto slab_it = std::find_if(base_slabs.begin(), base_slabs.end(), [&](const TimeSlabSpan &slab) {
            return slab.slab_id == assignment.slab_id;
        });
        const std::size_t base_index = static_cast<std::size_t>(slab_it - base_slabs.begin());
        const std::uint32_t output = base_to_output[base_index];
        plan.input_rows[plan.slabs[output].input_count++] = assignment.input_row;
    }
    for (std::size_t slot = 0u; slot < plan.slabs.size(); ++slot) {
        const std::uint32_t end = slot + 1u < plan.slabs.size()
            ? plan.slabs[slot + 1u].input_begin
            : total;
        plan.slabs[slot].input_count = end - plan.slabs[slot].input_begin;
    }

    return plan;
}

} // namespace cellerator::compute::graph
