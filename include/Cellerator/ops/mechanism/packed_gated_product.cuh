#pragma once

#include <Cellerator/ops/mechanism/gated_product.hh>
#include <Cellerator/tensor/block_masked/structure.hh>
#include <Cellerator/tensor/block_masked/view.hh>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cellerator::ops::mechanism {

// All pointed-to numeric/metadata arrays are device-local and caller-retained.
// Groups may be a local subset, with their original argument/output ranges.
// uint16_t storage means IEEE FP16 bits; float is the explicit FP32 alternative.
template<class Storage>
struct PackedGatedProductInput {
    tensor::block_masked::View<const Storage> x{};
    GatedProductTopologyView topology{};
    const std::uint64_t* argument_value_offsets = nullptr; // [batch, argument_count].
    const std::uint64_t* coordinate_value_offsets = nullptr; // [batch, coordinates]; include_baseline.
    const float* z = nullptr; // [batch, conditions, 2].
    const float* theta = nullptr;
    const float* phi = nullptr;
    const float* tau = nullptr;
    std::uint64_t batch = 0;
    float beta = 0.25f;
    bool include_baseline = true; // Ownership is explicit, independent of beta's numerical value.
};

struct PackedGatedProductSaved {
    float* gate_arguments = nullptr; // [batch, local group_count], both or neither.
    float* products = nullptr;
};

struct PackedGatedProductGradients {
    float* x = nullptr;     // [batch, coordinates], canonical coordinate order.
    float* z = nullptr;     // [batch, conditions, 2].
    float* theta = nullptr; // [parameters].
    float* phi = nullptr;
    float* tau = nullptr;
};

// Accessible byte extents of the corresponding borrowed buffers, supplied by
// their actual owners. The checked boundary rejects short and overlapping
// writable ranges before enqueueing any CUDA work.
struct PackedGatedProductCapacities {
    std::size_t values = 0, groups = 0, arguments = 0, outputs = 0;
    std::size_t argument_value_offsets = 0, coordinate_value_offsets = 0;
    std::size_t z = 0, theta = 0, phi = 0, tau = 0;
    std::size_t output = 0, output_gradient = 0;
    std::size_t gradient_x = 0, gradient_z = 0;
    std::size_t gradient_theta = 0, gradient_phi = 0, gradient_tau = 0;
    std::size_t gate_arguments = 0, products = 0, status = 0;
};

enum PackedGatedProductStatus : std::uint32_t {
    packed_invalid_topology = 1u,
    packed_missing_value = 2u,
    packed_nonfinite = 4u
};

// Host preparation only. Structure rows/features are mapped back to canonical
// coordinates explicitly. Every declared participant requires an allocated slot,
// including x == 0. Unused ranges of a group-subset topology get UINT64_MAX.
std::vector<std::uint64_t> prepare_argument_value_offsets(
    const tensor::block_masked::Structure&, GatedProductTopologyView host_topology);
std::vector<std::uint64_t> prepare_coordinate_value_offsets(
    const tensor::block_masked::Structure&, std::uint64_t coordinates);
// Explicit batch order supports partition/global row IDs and subsets. The
// overloads above use the Structure's row IDs in ascending canonical order.
std::vector<std::uint64_t> prepare_argument_value_offsets(
    const tensor::block_masked::Structure&, GatedProductTopologyView,
    const std::vector<std::uint64_t>& batch_rows);
std::vector<std::uint64_t> prepare_coordinate_value_offsets(
    const tensor::block_masked::Structure&, std::uint64_t coordinates,
    const std::vector<std::uint64_t>& batch_rows);

// Low-level unchecked buffer-capacity entrypoints for already admitted bindings.
// Output and gradients ACCUMULATE; initialize them before a logical evaluation.
// Caller also initializes the device status word to zero. Validation is queued
// before arithmetic. An arithmetic error may leave partial accumulations; any
// nonzero status makes those buffers unpublishable. Inspect status before
// adopting outputs. No allocation, transfer or
// host synchronization occurs. Inactive modules read neither x nor theta.
// With a saved pair, backward reads retained gate arguments/products and active
// participants, theta, phi, z1 and output gradients; it never reads z0 or tau.
// Without a saved pair, standalone backward recomputes the gate from z0,z1,phi,
// tau and therefore requires those numerical inputs too. Baseline backward
// reads only output gradients. Both modes retain the same complete allocation
// descriptor/capacity contract; an unused allocation is not a numerical read.
template<class Storage>
cudaError_t launch_packed_forward(
    PackedGatedProductInput<Storage> input, float* output,
    PackedGatedProductSaved saved, std::uint32_t* status, cudaStream_t stream) noexcept;

template<class Storage>
cudaError_t launch_packed_backward(
    PackedGatedProductInput<Storage> input, const float* output_gradient,
    PackedGatedProductGradients gradients, PackedGatedProductSaved saved,
    std::uint32_t* status, cudaStream_t stream) noexcept;

template<class Storage>
cudaError_t launch_packed_forward_checked(
    PackedGatedProductInput<Storage>, PackedGatedProductCapacities,
    float* output, PackedGatedProductSaved, std::uint32_t* status, cudaStream_t) noexcept;
template<class Storage>
cudaError_t launch_packed_backward_checked(
    PackedGatedProductInput<Storage>, PackedGatedProductCapacities,
    const float* output_gradient, PackedGatedProductGradients,
    PackedGatedProductSaved, std::uint32_t* status, cudaStream_t) noexcept;

// Pure host admission: examines only descriptors, accessible byte counts and
// address ranges. It does not dereference device data, call CUDA, or enqueue a
// readiness wait. Prepared programs use this before any per-launch work.
template<class Storage>
bool validate_packed_forward_buffers(
    PackedGatedProductInput<Storage>, PackedGatedProductCapacities,
    float* output, PackedGatedProductSaved, std::uint32_t* status) noexcept;
template<class Storage>
bool validate_packed_backward_buffers(
    PackedGatedProductInput<Storage>, PackedGatedProductCapacities,
    const float* output_gradient, PackedGatedProductGradients,
    PackedGatedProductSaved, std::uint32_t* status) noexcept;

extern template cudaError_t launch_packed_forward<std::uint16_t>(
    PackedGatedProductInput<std::uint16_t>, float*, PackedGatedProductSaved,
    std::uint32_t*, cudaStream_t) noexcept;
extern template cudaError_t launch_packed_forward<float>(
    PackedGatedProductInput<float>, float*, PackedGatedProductSaved,
    std::uint32_t*, cudaStream_t) noexcept;
extern template cudaError_t launch_packed_backward<std::uint16_t>(
    PackedGatedProductInput<std::uint16_t>, const float*, PackedGatedProductGradients,
    PackedGatedProductSaved, std::uint32_t*, cudaStream_t) noexcept;
extern template cudaError_t launch_packed_backward<float>(
    PackedGatedProductInput<float>, const float*, PackedGatedProductGradients,
    PackedGatedProductSaved, std::uint32_t*, cudaStream_t) noexcept;

} // namespace cellerator::ops::mechanism
