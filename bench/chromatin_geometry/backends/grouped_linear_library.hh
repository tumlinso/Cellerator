#pragma once

// Fixed-batch dense library control for the chromatin-geometry study.
// This is private experiment code, not an installed Cellerator API.
#include "../contract.hh"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cgs { struct ConsumerHostFixture; }

namespace cgs::backends {

struct LinearLibraryGroup {
    std::uint64_t parameter_index = 0;
    std::uint64_t condition_index = 0;
    std::uint64_t logical_id = 0;
    // Ordered slots are retained, including repeated participants.
    std::vector<std::uint64_t> arguments;
    // One incidence per local output feature; duplicate coordinates are valid.
    std::vector<cellerator::ops::mechanism::OutputIncidence> outputs;
};

// Immutable host preparation copied from the exact fixture topology. The
// library control deliberately requires uniform I/O dimensions because the
// existing GroupedLinear/cuBLAS path has one matrix shape per executor. A
// caller with other shapes buckets them into a separate control or rejects the
// library route; it must not pad a different numerical contract silently.
struct LinearLibraryLayout {
    std::uint64_t batch = 0;
    std::uint64_t coordinates = 0;
    std::uint64_t conditions = 0;
    std::uint64_t argument_count = 0;
    std::uint64_t weight_count = 0;
    std::uint64_t group_weight_count = 0;
    std::uint64_t parameter_count = 0;
    std::uint64_t structure_epoch = 0;
    std::uint64_t mapping_revision = 0;
    int device = 0;
    std::vector<std::uint64_t> parameter_weight_offsets; // dtype elements
    std::vector<std::uint64_t> group_weight_offsets;     // dtype elements
    std::vector<LinearLibraryGroup> groups;
};

enum class LinearLibraryArm : std::uint8_t {
    per_group,
    batched
};

// Uses the same fixed-B cuBLAS GEMM shapes, accumulation type, and rounding
// contract as GroupedLinear. The per_group arm makes one GEMM call for each
// expert found active in the current gate snapshot; the batched arm groups
// those calls by direction. The model's weight allocation must keep the same
// address for the lifetime of the executor; value updates in place are observed
// normally. Construction owns descriptor copies, activity staging, and cuBLAS setup,
// so include it in cold preparation cost. Methods enqueue numerical work on the
// caller's stream and allocate nothing, but each call copies the activity summary
// to pinned host memory and synchronizes that stream before selecting active
// experts. Include this readback and synchronization in measured execution cost.
// Calls on one executor must remain serial and forward/backward must use the same
// stream and exact gate revisions so saved rounded F values are consumed in order.
// `batched` packs the same active fixed-B GEMMs into cublasGemmBatchedEx calls.
// Its pointer-array construction and transfer are part of the measured call.
class GroupedLinearLibraryControl final {
public:
    GroupedLinearLibraryControl(LinearLibraryLayout layout,
                                const ProblemView& problem, DType dtype,
                                LinearLibraryArm arm = LinearLibraryArm::per_group);
    ~GroupedLinearLibraryControl();
    GroupedLinearLibraryControl(const GroupedLinearLibraryControl&) = delete;
    GroupedLinearLibraryControl& operator=(const GroupedLinearLibraryControl&) = delete;
    GroupedLinearLibraryControl(GroupedLinearLibraryControl&&) = delete;
    GroupedLinearLibraryControl& operator=(GroupedLinearLibraryControl&&) = delete;

    std::size_t workspace_bytes() const noexcept;
    // Requested bytes owned by this control: internal device descriptors and
    // staging buffers, plus pinned host staging. These exclude caller-owned
    // numerical workspace and opaque cuBLAS handle/library allocations.
    std::size_t device_bytes() const noexcept;
    std::size_t pinned_host_bytes() const noexcept;
    const LinearLibraryLayout& layout() const noexcept;
    LinearLibraryArm arm() const noexcept;

    cudaError_t forward(const ProblemView& problem, const GateView& gate,
                        ResultView result, WorkspaceView workspace,
                        cudaStream_t stream) noexcept;
    cudaError_t backward(const ProblemView& problem, const GateView& gate,
                         const float* dy, ResultView result,
                         WorkspaceView workspace,
                         cudaStream_t stream) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Host fixture adapter and poison-read correctness probe used by the study's
// admitted Scheduler validation job. The probe poisons D\P participant
// values through forward, restores the declared values, then checks the
// backward recomputation and STE dq against the independent scalar oracle.
LinearLibraryLayout make_linear_library_layout(
    const cgs::ConsumerHostFixture& fixture, int device);
bool run_grouped_linear_library_tests(cudaStream_t stream,
                                      std::string* error = nullptr);

} // namespace cgs::backends
