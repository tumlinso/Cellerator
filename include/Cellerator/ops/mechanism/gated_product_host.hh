#pragma once

#include "Cellerator/ops/mechanism/gated_product.hh"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace cellerator::ops::mechanism {

namespace detail {
template<class T>
struct GatedProductHostTapeState;
struct GatedProductHostTapeAccess;
} // namespace detail

// A one-shot, owning forward tape for the host reference implementation.
// It snapshots the prepared topology and every primal needed by backward, so
// callers may release or mutate their source buffers after forward returns.
// Backward consumes an rvalue tape. A successful call can be made once;
// malformed cotangent extents are rejected before the tape is consumed.
template<class T>
class GatedProductHostTape {
public:
    GatedProductHostTape(const GatedProductHostTape&) = delete;
    GatedProductHostTape& operator=(const GatedProductHostTape&) = delete;
    GatedProductHostTape(GatedProductHostTape&&) noexcept;
    GatedProductHostTape& operator=(GatedProductHostTape&&) noexcept;
    ~GatedProductHostTape();

    [[nodiscard]] bool consumed() const noexcept;
    // Counts argument slots numerically read by active module terms in
    // forward. Baseline reads are independent and are not counted here.
    [[nodiscard]] std::size_t forward_participant_slot_reads() const noexcept;

private:
    explicit GatedProductHostTape(
        std::unique_ptr<detail::GatedProductHostTapeState<T>> state) noexcept;

    std::unique_ptr<detail::GatedProductHostTapeState<T>> state_;

    friend struct detail::GatedProductHostTapeAccess;
};

template<class T>
struct GatedProductHostForward {
    std::vector<T> values;
    GatedProductHostTape<T> tape;
};

template<class T>
struct GatedProductHostBackward {
    std::vector<T> dx;
    std::vector<T> dz;
    std::vector<T> dtheta;
    std::vector<T> dphi;
    std::vector<T> dtau;

    // Number of argument slots read while evaluating active module VJPs.
    // This is useful for verifying that an inactive gate bypasses participants.
    std::size_t participant_slot_reads = 0;
};

// x is [batch, coordinate], z is [batch, condition, (z0,z1)] with the two
// condition channels interleaved, and each parameter span is [parameter].
// beta is a fixed scalar coefficient; it is not a trainable parameter.
// Topology and exact span extents are validated before an output is returned.
// Throws std::invalid_argument for malformed topology/shapes and
// std::overflow_error when a required extent cannot be represented.
[[nodiscard]] GatedProductHostForward<float> gated_product_forward(
    const GatedProductTopologyView& topology, std::size_t batch,
    std::span<const float> x, std::span<const float> z,
    std::span<const float> theta, std::span<const float> phi,
    std::span<const float> tau, float beta);

[[nodiscard]] GatedProductHostForward<double> gated_product_forward(
    const GatedProductTopologyView& topology, std::size_t batch,
    std::span<const double> x, std::span<const double> z,
    std::span<const double> theta, std::span<const double> phi,
    std::span<const double> tau, double beta);

// dvalues has the same [batch, coordinate] shape as the forward result.
// Every active argument occurrence contributes separately to dx; repeated
// indices and repeated output incidences therefore accumulate by occurrence.
// Gradients for shared parameters and conditions accumulate across batches and
// groups. The baseline contributes -beta*dvalues to dx exactly once.
[[nodiscard]] GatedProductHostBackward<float> gated_product_backward(
    GatedProductHostTape<float>&& tape, std::span<const float> dvalues);

[[nodiscard]] GatedProductHostBackward<double> gated_product_backward(
    GatedProductHostTape<double>&& tape, std::span<const double> dvalues);

} // namespace cellerator::ops::mechanism
