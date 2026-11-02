#pragma once

// Independent scalar case data and equations for native-data tests. This file
// intentionally depends only on the C++ standard library, not on candidate
// layouts, kernels, or execution code.
#include <array>
#include <cstddef>
#include <cstdint>

namespace cellerator::test::native_data_reference {

struct mechanism {
    std::array<std::size_t, 3> ordered_inputs{};
    std::size_t arity = 0;
    std::array<double, 5> output_scales{};
    std::uint64_t logical_id = 0;
    std::uint64_t parameter_index = 0;
    std::uint64_t condition_index = 0;
    int device = 0;
};

struct values {
    std::array<double, 5> x{};
    double beta = 0.0;
    double theta = 0.0;
    double phi = 0.0;
    double tau = 0.0;
    double z0 = 0.0;
    double z1 = 0.0;
};

inline constexpr std::array<mechanism, 4> mechanisms{{
    {{{0, 1, 1}}, 3, {{1.0, 0.0, 0.5, 0.0, 0.0}}, 101, 0, 0, 0},
    {{{1, 2, 0}}, 2, {{2.0, 0.0, 0.0, 0.0, 0.0}}, 102, 0, 0, 1},
    {{{3, 4, 0}}, 2, {{1.0, 0.0, 0.0, 0.0, 0.0}}, 103, 0, 0, 1},
    {{{3, 3, 0}}, 2, {{0.25, 0.0, 0.0, 0.0, 0.0}}, 104, 0, 0, 1},
}};

inline constexpr std::array<double, 5> loss_seed{{1.0, 0.0, -0.5, 0.0, 0.0}};
inline constexpr std::array<std::uint64_t, 5> coordinate_ids{{1001, 1002, 1003, 1004, 1005}};
inline constexpr std::array<std::uint64_t, 5> output_ids{{2001, 2002, 2003, 2004, 2005}};
inline constexpr std::uint64_t z0_condition_owner_device = 1;
inline constexpr std::uint64_t shared_parameter_owner_device = 0;

inline constexpr values active_values{
    {{2.0, 3.0, 0.5, 0.0, 2.5}}, // x0, x1, x2, x3, x4
    0.25,                         // baseline coefficient beta
    0.4,                          // shared parameter theta
    0.5,                          // gate parameter phi
    0.5,                          // gate threshold tau
    1.0,                          // regulator source z0, owned on device 1
    0.5,                          // condition z1
};

inline constexpr values inactive_values{
    active_values.x,
    active_values.beta,
    active_values.theta,
    0.5,
    0.6,
    0.1,
    0.5,
};

inline constexpr double gate_argument(const values& v) noexcept {
    return v.z0 + v.phi * v.z1 - v.tau;
}

// The first-order gate has a certified flat inactive branch. The reader is
// invoked only inside that branch's active path, after its condition is known.
// Baseline output terms are independent and are evaluated exactly once by the
// caller that owns the outputs.
template<class ParticipantReader>
std::array<double, 5> evaluate(const values& v, ParticipantReader&& read,
                               int selected_device = -1,
                               bool include_baseline = true) {
    std::array<double, 5> y{};
    if (include_baseline) {
        for (std::size_t i = 0; i < y.size(); ++i)
            y[i] = -v.beta * v.x[i];
    }

    const double s = gate_argument(v);
    if (s <= 0.0)
        return y;

    const double gated_scale = v.theta * s * s;
    for (const auto& group : mechanisms) {
        if (selected_device >= 0 && group.device != selected_device)
            continue;
        double product = 1.0;
        for (std::size_t slot = 0; slot < group.arity; ++slot)
            product *= read(group.ordered_inputs[slot]);
        for (std::size_t out = 0; out < y.size(); ++out)
            y[out] += group.output_scales[out] * gated_scale * product;
    }
    return y;
}

inline constexpr std::array<double, 5> expected_outputs{{4.225, -0.75, 1.9, 0.0, -0.625}};
inline constexpr std::array<double, 5> expected_dx{{1.26875, 2.25, 1.475, 0.5625, 0.0}};
inline constexpr std::array<double, 5> expected_device0_dx{{1.51875, 2.025, 0.0, 0.0, 0.0}};
inline constexpr std::array<double, 5> expected_device1_dx{{0.0, 0.225, 1.35, 0.5625, 0.0}};
inline constexpr std::array<double, 5> expected_dtheta_dz0_dz1_dphi_dtau{{
    9.28125, 9.9, 4.95, 4.95, -9.9,
}};
inline constexpr std::array<double, 5> expected_device0_dtheta_dz0_dz1_dphi_dtau{{
    7.59375, 8.1, 4.05, 4.05, -8.1,
}};
inline constexpr std::array<double, 5> expected_device1_dtheta_dz0_dz1_dphi_dtau{{
    1.6875, 1.8, 0.9, 0.9, -1.8,
}};

inline constexpr double expected_theta_after_one_sgd_update = 0.3071875;
inline constexpr double expected_y0_after_update = 3.12865234375;
inline constexpr double expected_y2_after_update = 1.43013671875;

} // namespace cellerator::test::native_data_reference
