#pragma once

#include "../contract.hh"
#include "../workloads.hh"
#include <memory>
#include <string>
#include <vector>

namespace cgs {

// Host fixture storage, also used by the benchmark runner. Returned views are
// borrowed and must be rebound to device allocations before a CUDA launch.
struct ConsumerHostFixture {
    BodyKind body = BodyKind::ordered_product;
    DType dtype = DType::fp16;
    GateProfile profile = GateProfile::supplied;
    std::uint64_t batch=0, coordinates=0, conditions=0, parameters=0;
    std::vector<cellerator::ops::mechanism::GatedProductGroup> groups;
    std::vector<std::uint64_t> arguments;
    std::vector<cellerator::ops::mechanism::OutputIncidence> outputs;
    std::vector<std::uint16_t> x16,weights16;
    std::vector<float> x32,weights32,theta,probabilities,dy;
    std::vector<std::uint8_t> admitted;
    std::vector<std::uint32_t> primal,derivative;
    std::vector<std::uint64_t> weight_offsets,group_weight_offsets;
    std::vector<std::uint64_t> neighborhood_group_offsets,neighborhood_groups;
    std::vector<std::uint64_t> neighborhood_feature_offsets,neighborhood_features;
    std::vector<std::uint32_t> feature_group_masks,argument_local_indices;
    std::uint64_t max_groups=0,max_features=0,max_group_weights=0;
    GateRevisions revisions{};

    ProblemView problem() const;
    GateView gates() const;
};

// Numerical consumers are controlled fixtures. Their participant/output IDs
// come from the supplied workload neighborhoods, including distal members.
// Linear fixtures use at most eight input and four output slots per group;
// product fixtures retain the complete ordered group membership.
ConsumerHostFixture make_consumer_fixture(const WorkloadFixture&,BodyKind,
                                           DType,GateProfile);

struct ConsumerResults {
    std::vector<float> y,dx,dtheta,dq,dweights,dx_group,dweights_group;
    std::uint32_t status=0;
};

// Allocation/transfer helper shared by correctness and timing runners. Every
// method that touches the device is restricted to an admitted Scheduler job.
// Constructor and download synchronize explicitly; timed kernels do not.
class ConsumerDeviceFixture {
public:
    ConsumerDeviceFixture(const ConsumerHostFixture&,cudaStream_t);
    ~ConsumerDeviceFixture();
    ConsumerDeviceFixture(const ConsumerDeviceFixture&)=delete;
    ConsumerDeviceFixture& operator=(const ConsumerDeviceFixture&)=delete;
    ProblemView problem() const;
    GateView gates() const;
    ResultView results() const;
    const float* dy() const;
    WorkspaceView workspace(Backend,BackendOptions);
    void clear_results(cudaStream_t);
    ConsumerResults download(cudaStream_t) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Admitted GPU job only. Includes tails, arity32/33, repeated/zero slots,
// shared output/parameter reductions, dormant STE dq and stale descriptors.
bool run_consumer_tests(cudaStream_t,std::string* error);
// Small supplied-product qualification for the grouping-payoff probe.
bool run_grouping_probe_qualification(cudaStream_t,std::string* error);
} // namespace cgs
