#pragma once

#include "../contract.hh"
#include "../workloads.hh"
#include <memory>
#include <string>

namespace cgs {
enum class ConditionProducer : std::uint8_t { root_mapped, bitplanes, bitmap_api };

// Immutable DAG/identity metadata is prepared and uploaded by the constructor.
// Both producer paths execute the same actual nodes; they do not consume
// precomputed host root decisions. Updates compare before/after leaf inputs
// and propagate cell-local/word-local dirty state through the actual DAG.
// root_mapped writes root bytes directly from shared node state. It does not
// create row bitmaps or condition planes unless make_planes is explicitly true.
// bitplanes always emits planes; bitmap_api retains the existing conversion
// control as an additional arm. Returned GateView pointers are null for an
// unrequested representation.
class ConditionDeviceFixture {
public:
    ConditionDeviceFixture(const WorkloadFixture&,cudaStream_t);
    ~ConditionDeviceFixture();
    ConditionDeviceFixture(const ConditionDeviceFixture&)=delete;
    ConditionDeviceFixture& operator=(const ConditionDeviceFixture&)=delete;
    cudaError_t rebuild(ConditionProducer,bool after,GateProfile,bool make_bytes,
                        cudaStream_t,bool make_planes=false) noexcept;
    // Call only after rebuilding with the same producer. Each call toggles
    // before/after accessibility, enabling repeated real leaf updates without
    // a hidden full rebuild or resetting cached numerical state.
    cudaError_t update(ConditionProducer,GateProfile,bool make_bytes,
                       cudaStream_t,bool make_planes=false) noexcept;
    GateView gates(GateProfile) const;
    std::uint32_t status(cudaStream_t) const;
    const std::uint32_t* device_status() const;
    std::size_t device_bytes() const;
    std::uint64_t evaluated_node_count() const;
    std::uint64_t source_node_count() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};

bool run_condition_tests(cudaStream_t,std::string* error);
} // namespace cgs
