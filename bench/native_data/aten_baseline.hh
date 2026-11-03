#pragma once

#include "workload.hh"

#include <cstdint>
#include <memory>
#include <vector>

namespace cellerator::bench::native_data {

struct TimingBreakdown {
    double host_pack_ms = 0.0;
    double upload_ms = 0.0;
    double readiness_ms = 0.0;
    double forward_ms = 0.0;
    double backward_ms = 0.0;
    double reduction_ms = 0.0;
    double maintenance_ms = 0.0;
    double update_ms = 0.0;
    double replica_refresh_ms = 0.0;
    double canonicalization_ms = 0.0;
    double validation_export_ms = 0.0;
    double full_cycle_ms = 0.0;
    std::uint64_t transfer_bytes = 0;
    std::uint64_t peak_allocated_bytes = 0;
    std::uint64_t peak_reserved_bytes = 0;
    std::uint64_t condition_transfer_bytes = 0;
    std::uint64_t participant_transfer_bytes = 0;
    std::uint64_t parameter_transfer_bytes = 0;
    std::uint64_t output_transfer_bytes = 0;
    std::uint64_t backward_transfer_bytes = 0;
    std::uint64_t replica_refresh_bytes = 0;
    std::uint64_t prepared_device_bytes = 0;
    std::uint64_t prepared_staging_bytes = 0;
    std::uint64_t prepared_host_payload_bytes = 0;
    std::uint64_t omitted_participant_transfers = 0;
    std::uint64_t projection_refreshes = 0;
    std::uint64_t projected_participant_values = 0;
    bool peer_forward = false;
    bool peer_reverse = false;
    bool update_includes_replica_refresh = false;
    std::uint64_t active_groups = 0;
    double gate_argument_min = 0.0;
    double gate_argument_max = 0.0;
};

struct CycleResult {
    TimingBreakdown timing;
    // Filled only when collect_values is true. VJPs use the Workload's
    // canonical row-major order, and theta_after_update is captured after the
    // deterministic owner-device SGD step and replica refresh.
    std::vector<float> output;
    std::vector<float> dx;
    std::vector<float> dz;
    std::vector<float> dtheta;
    std::vector<float> dphi;
    std::vector<float> dtau;
    std::vector<float> theta_after_update;
};

// Conventional ATen eager decomposition of the exact gated ordered-product
// contract. The topology is partitioned over two CUDA devices; dense inputs
// and shared parameters are replicated, partial outputs/VJPs are reduced to
// device 0, and one explicit SGD update is refreshed to device 1. It uses
// ATen gather/prod/index_add/autograd rather than a Cellerator kernel.
class AtenTwoDeviceBaseline {
public:
    AtenTwoDeviceBaseline(const Workload& workload, int device0, int device1);
    ~AtenTwoDeviceBaseline();
    AtenTwoDeviceBaseline(AtenTwoDeviceBaseline&&) noexcept;
    AtenTwoDeviceBaseline& operator=(AtenTwoDeviceBaseline&&) noexcept;
    AtenTwoDeviceBaseline(const AtenTwoDeviceBaseline&) = delete;
    AtenTwoDeviceBaseline& operator=(const AtenTwoDeviceBaseline&) = delete;

    const TimingBreakdown& preparation() const noexcept;
    CycleResult cycle(float learning_rate, bool update_parameters,
                      bool collect_values = false);
    CycleResult forward_only(bool collect_values = false);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cellerator::bench::native_data
