#pragma once

#include "../../compute/autograd/autograd.hh"
#include "../../../extern/CellShard/include/CellShard/runtime/device/sharded_device.cuh"

#include <cstdint>

namespace cellerator::models::developmental_time {

namespace autograd = ::cellerator::compute::autograd;
namespace csv = ::cellshard::device;

enum class DevelopmentalTimeLayout {
    blocked_ell,
    sliced_ell
};

struct DevelopmentalTimeBatchView {
    DevelopmentalTimeLayout layout = DevelopmentalTimeLayout::blocked_ell;
    std::uint32_t rows = 0u;
    csv::blocked_ell_view blocked_ell{};
    csv::sliced_ell_view sliced_ell{};
    const float *target_time = nullptr;
    const std::uint32_t *cell_index = nullptr;
};

inline DevelopmentalTimeBatchView make_developmental_time_blocked_ell_batch(
    const csv::blocked_ell_view &view,
    const float *target_time = nullptr,
    const std::uint32_t *cell_index = nullptr) {
    return DevelopmentalTimeBatchView{
        DevelopmentalTimeLayout::blocked_ell,
        view.rows,
        view,
        csv::sliced_ell_view{},
        target_time,
        cell_index
    };
}

inline DevelopmentalTimeBatchView make_developmental_time_sliced_ell_batch(
    const csv::sliced_ell_view &view,
    const float *target_time = nullptr,
    const std::uint32_t *cell_index = nullptr) {
    return DevelopmentalTimeBatchView{
        DevelopmentalTimeLayout::sliced_ell,
        view.rows,
        csv::blocked_ell_view{},
        view,
        target_time,
        cell_index
    };
}

using TimeBatch = DevelopmentalTimeBatchView;

} // namespace cellerator::models::developmental_time
