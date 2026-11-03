#pragma once

// Study-private ABI. All device pointers are borrowed, device-local, and must
// remain valid through completion on the supplied CUDA stream. This is not an
// installed Cellerator interface.
#include <Cellerator/ops/mechanism/gated_product.hh>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cgs {

enum class DType : std::uint8_t { fp16, fp32 };
enum class GateProfile : std::uint8_t { supplied, ste, teacher };
enum class BodyKind : std::uint8_t { ordered_product, grouped_linear };
enum class Backend : std::uint8_t {
    scalar, bitmap, direct, cohort_direct, cohort_shared, cohort_direct_two_tile,
    masked_direct_two_tile
};

// Revisions describe the inputs/meaning of a saved gate or schedule; they do
// not retain source bytes, establish stream readiness, or replace ownership.
// T and threshold are recorded as their exact IEEE-754 binary32 bit patterns.
struct GateRevisions {
    std::uint64_t structure_epoch = 0;
    std::uint64_t mapping_revision = 0;
    std::uint64_t accessibility_generation = 0;
    std::uint64_t score_generation = 0;
    std::uint64_t parameter_revision = 0;
    std::uint64_t teacher_revision = 0;
    std::uint64_t temporal_generation = 0;
    std::uint64_t primal_generation = 0;
    std::uint64_t derivative_generation = 0;
    std::uint32_t temperature_bits = 0;
    std::uint32_t threshold_bits = 0;
};

inline constexpr bool same_revisions(const GateRevisions& a,
                                     const GateRevisions& b) noexcept {
    return a.structure_epoch == b.structure_epoch &&
           a.mapping_revision == b.mapping_revision &&
           a.accessibility_generation == b.accessibility_generation &&
           a.score_generation == b.score_generation &&
           a.parameter_revision == b.parameter_revision &&
           a.teacher_revision == b.teacher_revision &&
           a.temporal_generation == b.temporal_generation &&
           a.primal_generation == b.primal_generation &&
           a.derivative_generation == b.derivative_generation &&
           a.temperature_bits == b.temperature_bits &&
           a.threshold_bits == b.threshold_bits;
}

// A work partition only: every group occurs once in groups, but participating
// coordinates and outputs may overlap neighborhoods or be genomically distal.
// The feature union is an address map, not permission to load every feature.
// Bit j in feature_group_masks identifies a local group using that feature.
// Phase-specific demand is the intersection with its primal or derivative
// signature; an inactive-only forward participant must never be loaded.
struct NeighborhoodView {
    const std::uint64_t* group_offsets = nullptr;   // [count+1]
    const std::uint64_t* groups = nullptr;          // [group_memberships]
    const std::uint64_t* feature_offsets = nullptr; // [count+1]
    const std::uint64_t* features = nullptr;        // [feature_memberships]
    const std::uint32_t* feature_group_masks = nullptr; // [feature_memberships]
    const std::uint32_t* argument_local_indices = nullptr; // [argument_count]
    std::uint64_t count = 0;
    std::uint64_t group_memberships = 0;
    std::uint64_t feature_memberships = 0;
    std::uint64_t max_groups = 0;
    std::uint64_t max_features = 0;
    std::uint64_t max_group_weights = 0; // sum of linear weights in one neighborhood
};

using ProductTopology = cellerator::ops::mechanism::GatedProductTopologyView;

struct ProblemView {
    ProductTopology topology{};
    std::uint64_t batch = 0;
    DType dtype = DType::fp16;
    const void* x = nullptr; // [batch,coordinates], uint16 FP16 bits or float
    const float* theta = nullptr; // [parameters], ordered-product only
    const void* weights = nullptr; // same dtype as x, grouped-linear only
    const std::uint64_t* weight_offsets = nullptr; // [parameters], into weights
    const std::uint64_t* group_weight_offsets = nullptr; // [groups+1], local dW
    std::uint64_t weight_count = 0;
    std::uint64_t group_weight_count = 0;
    NeighborhoodView neighborhoods{};
    float beta = 0.0f;
    bool include_baseline = false; // false permits no-read inactive fixtures
};

// Gate arrays retain distinct forward and derivative membership. Byte
// admission [batch,conditions] is the scalar-control representation. Plane
// indexing is [condition,ceil(batch/32)], bit cell%32. For STE probabilities
// are [batch,conditions], q=p when admitted and q=0 otherwise. For supplied
// and hardened teacher gates q=admitted and probabilities may be null.
struct GateView {
    GateProfile profile = GateProfile::supplied;
    const float* probabilities = nullptr;
    const std::uint8_t* admitted = nullptr;
    const std::uint32_t* primal_words = nullptr;
    const std::uint32_t* derivative_words = nullptr;
    std::uint64_t words_per_condition = 0;
    GateRevisions revisions{};
};

// Caller initializes accumulation destinations and retains all read inputs
// through backward completion. dweights_group and dx_group are dtype-sized
// local linear VJP results; linear GEMM results and VJPs round to dtype before
// any FP32 canonical overlap reduction. Product y/dx/dtheta/dq are FP32.
struct ResultView {
    float* y = nullptr;      // [batch,coordinates]
    float* dx = nullptr;     // [batch,coordinates], canonical
    float* dtheta = nullptr; // [parameters]
    float* dq = nullptr;     // [batch,conditions], STE shared-root accumulation;
                            // supplied/teacher gates leave caller-cleared dq zero
    float* dweights = nullptr; // [weight_count], canonical FP32 reduction
    void* dx_group = nullptr; // [batch,argument_count], dtype
    void* dweights_group = nullptr; // [group_weight_count], dtype
};

struct BackendOptions {
    std::uint32_t cell_tile = 32; // supported experimental choices 32/64/128
    std::uint32_t stage_count = 1; // supported experimental choices 1/2
};

enum StatusFlag : std::uint32_t {
    status_invalid_descriptor = 1u,
    status_unsupported_shape = 2u,
    status_insufficient_capacity = 4u,
    status_nonfinite = 8u,
    status_stale = 16u,
};

struct WorkspaceView {
    void* data = nullptr;
    std::size_t bytes = 0;
    std::uint32_t* status = nullptr; // device word; caller clears on stream
};

// Sizing uses only by-value host shape counts/maxima and never dereferences
// device pointers. The output is false for invalid or overflowing shapes.
bool workspace_bytes(Backend, BodyKind, const ProblemView&, BackendOptions,
                     std::size_t* bytes) noexcept;

// prepare owns conversion/cohort-queue construction costs. All calls enqueue
// on stream, allocate nothing, and synchronize neither host nor device. A
// nonzero device status makes that sequence unpublishable; host descriptors
// that can be rejected without reading device memory fail before enqueue.
cudaError_t prepare(Backend, BodyKind, const ProblemView&, const GateView&,
                    BackendOptions, WorkspaceView, cudaStream_t) noexcept;
cudaError_t forward(Backend, BodyKind, const ProblemView&, const GateView&,
                    BackendOptions, ResultView, WorkspaceView,
                    cudaStream_t) noexcept;
cudaError_t backward(Backend, BodyKind, const ProblemView&, const GateView&,
                     const float* dy, BackendOptions, ResultView, WorkspaceView,
                     cudaStream_t) noexcept;

static_assert(std::is_trivially_copyable_v<GateRevisions>);
static_assert(std::is_trivially_copyable_v<ProblemView>);
static_assert(std::is_trivially_copyable_v<GateView>);

} // namespace cgs
