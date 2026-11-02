#pragma once

#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#define CELLERATOR_GATED_PRODUCT_HD __host__ __device__
#else
#define CELLERATOR_GATED_PRODUCT_HD
#endif

namespace cellerator::ops::mechanism {

// Prepared coordinates are positions in explicitly bound logical axes. They
// are not biological identities or offsets in a particular physical tensor.
// Argument slots retain their order and multiplicity.
struct GatedProductGroup {
    std::uint64_t argument_begin = 0;
    std::uint64_t argument_count = 0;
    std::uint64_t output_begin = 0;
    std::uint64_t output_count = 0;
    std::uint64_t parameter_index = 0;
    std::uint64_t condition_index = 0;
    std::uint64_t logical_id = 0;
};

struct OutputIncidence {
    std::uint64_t coordinate = 0;
    float scale = 1;
};

struct GatedProductTopologyView {
    const GatedProductGroup* groups = nullptr;
    std::uint64_t group_count = 0;
    const std::uint64_t* arguments = nullptr;
    std::uint64_t argument_count = 0;
    const OutputIncidence* outputs = nullptr;
    std::uint64_t output_count = 0;
    std::uint64_t coordinates = 0;
    std::uint64_t conditions = 0;
    std::uint64_t parameters = 0;
};

template<class T>
struct GateEvaluation {
    T argument;
    T value;
    T derivative;
    bool excluded;
};

// The inactive branch is part of this model's declared numerical operation:
// no participant values are read there. It is not an IEEE rewrite of 0 * NaN.
// The squared positive part is C1, including its zero first derivative at 0;
// this contract makes no claim about second derivatives at that boundary.
template<class T>
CELLERATOR_GATED_PRODUCT_HD constexpr GateEvaluation<T> evaluate_gate(
    T z0, T z1, T phi, T tau) noexcept {
    const T argument = z0 + phi * z1 - tau;
    if (argument <= T{0}) return {argument, T{0}, T{0}, true};
    return {argument, argument * argument, T{2} * argument, false};
}

template<class T>
CELLERATOR_GATED_PRODUCT_HD constexpr T gated_contribution(
    T theta, const GateEvaluation<T>& gate, T product) noexcept {
    return gate.excluded ? T{0} : theta * gate.value * product;
}

template<class T>
struct GatedProductPartials {
    T participant_factor;
    T z0;
    T z1;
    T theta;
    T phi;
    T tau;
};

// participant_factor multiplies the product of all *other argument slots*.
// Consumers use prefix/suffix products, never product / x, so zeros and
// repeated occurrences retain their derivatives.
template<class T>
CELLERATOR_GATED_PRODUCT_HD constexpr GatedProductPartials<T> local_partials(
    T upstream, T theta, T phi, T z1, const GateEvaluation<T>& gate,
    T product) noexcept {
    if (gate.excluded) return {T{0}, T{0}, T{0}, T{0}, T{0}, T{0}};
    const T condition = upstream * theta * product * gate.derivative;
    return {upstream * theta * gate.value,
            condition, condition * phi, upstream * gate.value * product,
            condition * z1, -condition};
}

static_assert(std::is_trivially_copyable_v<GatedProductGroup>);
static_assert(std::is_trivially_copyable_v<OutputIncidence>);
static_assert(std::is_trivially_copyable_v<GatedProductTopologyView>);

} // namespace cellerator::ops::mechanism

#undef CELLERATOR_GATED_PRODUCT_HD
