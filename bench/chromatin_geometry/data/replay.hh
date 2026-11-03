#pragma once
#include "workloads.hh"
#include <cstdint>
#include <string>
namespace cgs {
struct ReplayPreparationTimes final {
    double operational_input_load_ms = 0.0;
    double reference_recompute_ms = 0.0;
    std::uint64_t payload_bytes_hashed = 0;
    std::uint64_t payload_bytes_read = 0;
};

// Reads the canonical, SHA-256-pinned 10x CGATAC01 payload. Selects cell and
// peak prefixes, applies the workload's declared input changes, and recomputes
// root observations. Non-detections are observations, never proven closures.
void apply_atac_replay(WorkloadFixture&, const std::string& payload_path);

// Optional host-side phase accounting for complete-cost studies. The input
// phase includes opening, hashing, validating, unpacking and applying declared
// leaf changes. The reference phase contains only CPU DAG reevaluation.
void apply_atac_replay(WorkloadFixture&, const std::string& payload_path,
                       ReplayPreparationTimes* timings);
}
