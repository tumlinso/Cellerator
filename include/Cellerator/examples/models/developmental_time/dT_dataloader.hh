#pragma once

#include <Cellerator/runtime/multi_gpu/fleet.cuh>
#include <Cellerator/tensor/base/device_views.cuh>

#include <cstdint>

namespace cellerator::models::developmental_time {

namespace runtime = ::cellerator::runtime;
namespace csv = ::cellerator::tensor::base::device;

enum class DevelopmentalTimeLayout {
    blocked_ell,
    sliced_ell
};

struct DevelopmentalTimeBatchView {
    DevelopmentalTimeLayout layout = DevelopmentalTimeLayout::blocked_ell;
    std::uint32_t rows = 0u;
    csv::BlockedEllView blocked_ell{};
    csv::SlicedEllView sliced_ell{};
    const float *target_time = nullptr;
    const std::uint32_t *cell_index = nullptr;
};

inline DevelopmentalTimeBatchView make_developmental_time_blocked_ell_batch(
    const csv::BlockedEllView &view,
    const float *target_time = nullptr,
    const std::uint32_t *cell_index = nullptr) {
    return DevelopmentalTimeBatchView{
        DevelopmentalTimeLayout::blocked_ell,
        view.rows,
        view,
        csv::SlicedEllView{},
        target_time,
        cell_index
    };
}

inline DevelopmentalTimeBatchView make_developmental_time_sliced_ell_batch(
    const csv::SlicedEllView &view,
    const float *target_time = nullptr,
    const std::uint32_t *cell_index = nullptr) {
    return DevelopmentalTimeBatchView{
        DevelopmentalTimeLayout::sliced_ell,
        view.rows,
        csv::BlockedEllView{},
        view,
        target_time,
        cell_index
    };
}

using TimeBatch = DevelopmentalTimeBatchView;

} // namespace cellerator::models::developmental_time
