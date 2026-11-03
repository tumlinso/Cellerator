#pragma once

#include <Cellerator/layout/factorized_interaction.hh>
#include <Cellerator/memory/cuda/resources.hh>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cellerator::modeling::cuda {

// First explicit law: g[m,c]=sigmoid(bias[b]+sum_k gamma[gk]*z[zk,c]);
// y[t,d,c] += g[m,c]*alpha[a,d]*product_r x[pr,d,c]. Input roles and
// condition terms are ordered, and duplicates contribute separate derivatives.
// This law is distinct from ops::mechanism's existing gated-product family.
struct FactorizedInteractionProblem {
    std::uint64_t cells = 0, channels = 0;
};

struct FactorizedInteractionPrecision {
    memory::cuda::DType activations = memory::cuda::DType::float16;
    memory::cuda::DType parameters = memory::cuda::DType::float16;
    memory::cuda::DType output = memory::cuda::DType::float16;
};

struct FactorizedInteractionValues {
    memory::cuda::ConstDeviceView x;     // [input occurrence, channel, cell]
    memory::cuda::ConstDeviceView z;     // [context occurrence, cell]
    memory::cuda::ConstDeviceView alpha; // [canonical alpha row, channel]
    memory::cuda::ConstDeviceView bias;  // [canonical bias scalar]
    memory::cuda::ConstDeviceView gamma; // [canonical gamma scalar]
};

// Native cotangent and every gradient are FP32, independently of primal/output
// storage. A caller adapter may explicitly cast gradients to its owner's dtype.
struct FactorizedInteractionGradients {
    memory::cuda::DeviceView x, z, alpha, bias, gamma;
};

enum class FactorizedInteractionRoute : std::uint8_t { packet, direct };
// Both requested routes share the accurate owner-reduction backend outside
// the conservative small-contribution atomic regime. Report this selection
// with benchmark results; the common fallback is not a packet-layout win.
enum class FactorizedInteractionBackend : std::uint8_t { packet, direct, stable };
enum class FactorizedInteractionGradientMode : std::uint8_t { overwrite, accumulate };

struct FactorizedInteractionWorkspace {
    std::size_t tape_bytes = 0;
    std::size_t forward_bytes = 0;
    std::size_t backward_bytes = 0;
};

// One immutable uploaded structural owner using the existing CUDA allocator.
// Construction validates host structure and queues one bulk upload. Same-stream
// use is ordered; another stream must call wait_on(). The owner must outlive all
// device uses of view(). It owns no numerical primals, parameters, or tapes.
class FactorizedInteractionDeviceLayout {
public:
    FactorizedInteractionDeviceLayout(
        const layout::FactorizedInteractionLayout& host, int device,
        const memory::cuda::Stream& upload_stream);
    ~FactorizedInteractionDeviceLayout() noexcept;
    FactorizedInteractionDeviceLayout(FactorizedInteractionDeviceLayout&&) noexcept;
    FactorizedInteractionDeviceLayout& operator=(FactorizedInteractionDeviceLayout&&) noexcept;
    FactorizedInteractionDeviceLayout(const FactorizedInteractionDeviceLayout&) = delete;
    FactorizedInteractionDeviceLayout& operator=(const FactorizedInteractionDeviceLayout&) = delete;
    const layout::FactorizedInteractionView& view() const noexcept;
    int device() const noexcept;
    std::size_t bytes() const noexcept;
    void wait_on(const memory::cuda::Stream& stream) const;
private:
    struct State;
    std::unique_ptr<State> state_;
};

// Pure host sizing. Returns cudaErrorInvalidValue for unsupported/overflowing
// dimensions. No CUDA context, allocation, synchronization or device reads.
cudaError_t factorized_interaction_workspace(
    const layout::FactorizedInteractionView& plan,
    FactorizedInteractionProblem problem, FactorizedInteractionRoute route,
    FactorizedInteractionWorkspace* result) noexcept;

// Pure host query using immutable structural maxima and the invocation shape.
cudaError_t factorized_interaction_backend(
    const layout::FactorizedInteractionView& plan,
    FactorizedInteractionProblem problem, FactorizedInteractionRoute route,
    FactorizedInteractionBackend* result) noexcept;

// Async, allocation-free execution. Every plan pointer must refer to immutable
// device storage on the current CUDA device, which must also match the supplied
// stream and all nonempty views. This API does not switch the current device.
// The plan must have
// passed host structural validation before upload. Buffers are disjoint except
// that read-only primal views may alias. Tape and scratch must be at least the
// queried byte extents, aligned for float; output is fully overwritten, and
// empty mechanisms produce zero. Empty cells/channels perform no numerical work.
//
// Tape stores two consecutive FP32 arrays [canonical mechanism, cell]: gates,
// then stable sigmoid slopes. The slope is exp(-abs(a))/(1+exp(-abs(a)))^2;
// retaining it avoids losing saturated gradients when the rounded gate is one.
// The caller retains plan,
// all immutable primals, tape and their existing owners through backward and
// queued stream completion. Tape is per invocation, not global scratch. Source
// mutation requires a new invocation; the low-level API cannot detect mutation
// or retain opaque borrowed owners. Forward/backward on another stream require
// an explicit caller event dependency. Scratch may be reused only after its
// previous queued use completes. Repeated VJPs are allowed with private scratch
// and gradient destinations. No implicit numerical snapshots or host transfers.
//
// FP32 joint products, gate arithmetic, reductions and gradients. Finite input
// values in the independently frozen qualification domain are the initial
// supported numerical contract; no threshold pruning or fast-math policy is
// applied. NaN/Inf propagation and overflow outside that domain are not a
// certified scientific approximation. Device structure is trusted after prep.
cudaError_t factorized_interaction_forward(
    const layout::FactorizedInteractionView& plan,
    FactorizedInteractionProblem problem, FactorizedInteractionPrecision precision,
    FactorizedInteractionValues values, memory::cuda::DeviceView output,
    memory::cuda::DeviceView tape, memory::cuda::DeviceView scratch,
    cudaStream_t stream, FactorizedInteractionRoute route = FactorizedInteractionRoute::packet) noexcept;

cudaError_t factorized_interaction_backward(
    const layout::FactorizedInteractionView& plan,
    FactorizedInteractionProblem problem, FactorizedInteractionPrecision precision,
    FactorizedInteractionValues values, memory::cuda::ConstDeviceView output_gradient,
    memory::cuda::ConstDeviceView tape, FactorizedInteractionGradients gradients,
    memory::cuda::DeviceView scratch, FactorizedInteractionGradientMode mode,
    cudaStream_t stream, FactorizedInteractionRoute route = FactorizedInteractionRoute::packet) noexcept;

// Direct-control symbols have the same borrowed/tape contract. Their bounded
// atomic implementation is separate and attributable; both routes explicitly
// report the same stable backend when accurate owner reductions are required.
cudaError_t factorized_interaction_direct_workspace(
    const layout::FactorizedInteractionView&, FactorizedInteractionProblem,
    FactorizedInteractionWorkspace*) noexcept;
cudaError_t factorized_interaction_direct_forward(
    const layout::FactorizedInteractionView&, FactorizedInteractionProblem,
    FactorizedInteractionPrecision, FactorizedInteractionValues,
    memory::cuda::DeviceView, memory::cuda::DeviceView, memory::cuda::DeviceView,
    cudaStream_t) noexcept;
cudaError_t factorized_interaction_direct_backward(
    const layout::FactorizedInteractionView&, FactorizedInteractionProblem,
    FactorizedInteractionPrecision, FactorizedInteractionValues,
    memory::cuda::ConstDeviceView, memory::cuda::ConstDeviceView,
    FactorizedInteractionGradients, memory::cuda::DeviceView,
    FactorizedInteractionGradientMode, cudaStream_t) noexcept;

} // namespace cellerator::modeling::cuda
