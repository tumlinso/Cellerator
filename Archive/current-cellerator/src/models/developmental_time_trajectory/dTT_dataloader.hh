#pragma once

#include "../developmental_time/dT_dataloader.hh"

namespace cellerator::models::developmental_time_trajectory {

namespace dt = ::cellerator::models::developmental_time;

struct DevelopmentalTimeGraphView {
    std::uint32_t edge_count = 0u;
    const std::uint32_t *src = nullptr;
    const std::uint32_t *dst = nullptr;
    const float *weight = nullptr;
};

struct DevelopmentalTimeTrajectoryBatchView {
    dt::DevelopmentalTimeBatchView features{};
    DevelopmentalTimeGraphView graph{};
};

inline DevelopmentalTimeTrajectoryBatchView make_developmental_time_trajectory_blocked_ell_batch(
    const dt::csv::blocked_ell_view &view,
    const float *target_time,
    DevelopmentalTimeGraphView graph,
    const std::uint32_t *cell_index = nullptr) {
    return DevelopmentalTimeTrajectoryBatchView{
        dt::make_developmental_time_blocked_ell_batch(view, target_time, cell_index),
        graph
    };
}

inline DevelopmentalTimeTrajectoryBatchView make_developmental_time_trajectory_sliced_ell_batch(
    const dt::csv::sliced_ell_view &view,
    const float *target_time,
    DevelopmentalTimeGraphView graph,
    const std::uint32_t *cell_index = nullptr) {
    return DevelopmentalTimeTrajectoryBatchView{
        dt::make_developmental_time_sliced_ell_batch(view, target_time, cell_index),
        graph
    };
}

using TimeTrajectoryBatch = DevelopmentalTimeTrajectoryBatchView;

} // namespace cellerator::models::developmental_time_trajectory
