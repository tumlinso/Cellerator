#pragma once

#include "aten_baseline.hh"

#include <array>
#include <memory>

namespace cellerator::bench::native_data {

// Benchmark adapter for the actual prepared two-device Cellerator consumer.
// This owns topology/layout preparation and delegates all numerical work,
// leases, transfers, reductions and updates to PreparedDistributedGatedProduct.
class DistributedGatedProductAdapter {
public:
    DistributedGatedProductAdapter(const Workload& workload, int device0, int device1);
    ~DistributedGatedProductAdapter();
    DistributedGatedProductAdapter(DistributedGatedProductAdapter&&) noexcept;
    DistributedGatedProductAdapter& operator=(DistributedGatedProductAdapter&&) noexcept;
    DistributedGatedProductAdapter(const DistributedGatedProductAdapter&) = delete;
    DistributedGatedProductAdapter& operator=(const DistributedGatedProductAdapter&) = delete;

    const TimingBreakdown& preparation() const noexcept;
    CycleResult cycle(float learning_rate, bool update_parameters,
                      bool collect_values = false);
    CycleResult forward_only(bool collect_values = false);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cellerator::bench::native_data
