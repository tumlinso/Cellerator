#include <Cellerator/ops/sparse_support/training.hh>

#define main ce_geo_mma_projection_roundtrip_regression
#include "../geometry/projection/mma_projection_roundtrip_test.cc"
#undef main

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace cellerator::compute::operation::sparse_support;

int main() {
    const auto image = physical_image();
    const std::uint64_t physical_edge_ids[] = {0, 1};
    const std::uint32_t physical_sources[] = {0, 0};
    const std::uint32_t physical_destinations[] = {0, 0};
    const std::uint32_t source_group_members[] = {0};
    const std::uint32_t destination_group_members[] = {0};
    ce_geo_n64_projection_binding_v1 binding;
    assert(prepare_ce_geo_n64_projection_v1(image.data(), image.size(),
        physical_edge_ids, physical_sources, physical_destinations,
        source_group_members, destination_group_members, 1, 1, 7, 11,
        &binding) == status_v1::success);
    assert(binding.edge_value_map_indices.size() == 2
        && binding.edge_value_map_indices[0] == 0
        && binding.edge_value_map_indices[1] == 1);
    assert(binding.tile_destination_groups[0] == 0
        && binding.tile_source_bases[0] == 0
        && binding.source_members.size() == 16
        && binding.source_members[0] == 0
        && binding.logical_source_to_physical[0] == 0
        && binding.destination_members[0] == 0);
    const std::uint64_t delta_edge_ids[] = {0, 1, 2};
    const std::uint32_t delta_sources[] = {0, 0, 1};
    const std::uint32_t delta_destinations[] = {0, 0, 1};
    assert(prepare_ce_geo_n64_projection_v1(image.data(), image.size(),
        delta_edge_ids, delta_sources, delta_destinations,
        source_group_members, destination_group_members, 2, 2, 7, 11,
        &binding) == status_v1::success);
    assert(binding.edge_value_map_indices[2]
        == std::numeric_limits<std::uint64_t>::max()
        && binding.logical_source_to_physical[1]
            == std::numeric_limits<std::uint32_t>::max());

    const std::uint64_t edge_ids[] = {5, 9};
    const std::uint32_t sources[] = {0, 1};
    const std::uint32_t destinations[] = {0, 0};
    edge_axis_view_v1 edges{3, 2, edge_ids, sources, destinations};
    const std::uint64_t cell_offsets[] = {0, 2};
    const std::uint64_t primal_ordinals[] = {0, 1};
    const std::uint64_t primal_ids[] = {5, 9};
    const std::uint64_t gate_index[] = {0, 1};
    per_cell_edge_workset_view_v1 primal{3, 4, 1, cell_offsets,
        primal_ordinals, primal_ids, gate_index, 2, 2};
    const std::uint64_t gate_ordinals[] = {0, 1};
    const std::uint64_t gate_ids[] = {5, 9};
    const float probabilities[] = {0.75f, 0.5f};
    const std::uint8_t admitted[] = {0, 1};
    edge_gate_values_view_v1 gates{3, 4, 1, gate_ordinals, gate_ids,
        probabilities, admitted, 2, 2};
    std::vector<float> x(2 * 64, 1.0f);
    std::fill(x.begin(), x.begin() + 64, std::numeric_limits<float>::quiet_NaN());
    std::fill(x.begin() + 64, x.end(), 2.0f);
    const float weights[] = {7.0f, 3.0f};
    std::vector<float> y(64);
    assert(sparse_forward_f32_reference_v1(edges, primal, gates, 2, 1,
        x, weights, y) == status_v1::success);
    for (float value : y) assert(value == 3.0f);

    const std::uint64_t derivative_ordinals[] = {1};
    const std::uint64_t derivative_ids[] = {9};
    const std::uint64_t derivative_gates[] = {1};
    const std::uint64_t derivative_offsets[] = {0, 1};
    per_cell_edge_workset_view_v1 derivative{3, 4, 1, derivative_offsets,
        derivative_ordinals, derivative_ids, derivative_gates, 1, 1};
    std::vector<float> dy(64, 1.0f), dx(128), dw(2), dg(1);
    assert(sparse_backward_f32_reference_v1(edges, primal, derivative, gates,
        2, 1, x, weights, dy, dx, dw, dg) == status_v1::success);
    assert(dw[0] == 0.0f && dw[1] == 64.0f);
    assert(dg[0] == 384.0f);
    for (std::size_t i = 0; i < 64; ++i) assert(dx[i] == 0.0f);
    for (std::size_t i = 64; i < dx.size(); ++i) assert(dx[i] == 1.5f);

    latent_transition_shape_v1 shape{2, 2, 2, 1};
    const std::uint32_t affected[] = {1};
    const std::uint64_t affected_offsets[] = {0, 1, 1};
    const float state[] = {0, 0, 1, -1, 2, 3, -2, 4};
    const float input[] = {0.25f, -0.5f};
    const float a[] = {1, 0, 0, 1};
    const float b[] = {2, -1};
    const float bias[] = {0, 0};
    std::vector<float> next(8), next_grad(8, 1.0f), state_grad(8),
        input_grad(2), da(4), db(2), dbias(2);
    assert(latent_transition_f32_reference_v1(shape, affected_offsets, affected, state, input,
        a, b, bias, next) == status_v1::success);
    assert(next[0] == state[0] && next[1] == state[1]
        && next[4] == state[4] && next[5] == state[5]
        && next[6] == state[6] && next[7] == state[7]);
    assert(std::abs(next[2] - std::tanh(1.5f)) < 1e-6f);
    assert(std::abs(next[3] - std::tanh(-1.25f)) < 1e-6f);
    assert(latent_transition_vjp_f32_reference_v1(shape, affected_offsets, affected, state,
        input, a, b, bias, next, next_grad, state_grad, input_grad, da, db,
        dbias) == status_v1::success);
    assert(state_grad[0] == 1.0f && state_grad[1] == 1.0f
        && state_grad[4] == 1.0f && state_grad[5] == 1.0f);
    assert(std::abs(state_grad[2] - (1.0f - next[2] * next[2])) < 1e-6f);
    assert(std::abs(state_grad[3] - (1.0f - next[3] * next[3])) < 1e-6f);

    // Independent double-precision scalar oracle for the additive transition
    // modes API, including all parameter and identity-path derivatives.
    const latent_transition_shape_v1 mode_shape{2, 3, 2, 2};
    const std::vector<std::uint64_t> mode_offsets{0, 2, 4};
    const std::vector<std::uint32_t> mode_affected{0, 2, 1, 2};
    std::vector<float> mode_state{.2f, -.4f, .5f, .1f, -.3f, .7f,
                                  .6f, -.2f, .4f, .3f, -.5f, .8f};
    std::vector<float> mode_input{.25f, -.75f, .5f, .125f};
    std::vector<float> mode_a(3 * 2 * 2), mode_b(3 * 2 * 2), mode_bias(3 * 2);
    for (std::size_t i = 0; i < mode_a.size(); ++i)
        mode_a[i] = .08f * static_cast<float>(i + 1) - .31f;
    for (std::size_t i = 0; i < mode_b.size(); ++i)
        mode_b[i] = .06f * static_cast<float>(i + 1) - .2f;
    for (std::size_t i = 0; i < mode_bias.size(); ++i)
        mode_bias[i] = .035f * static_cast<float>(i + 1) - .08f;
    std::vector<float> mode_upstream(mode_state.size());
    for (std::size_t i = 0; i < mode_upstream.size(); ++i)
        mode_upstream[i] = .17f * static_cast<float>(i + 1) - .6f;

    const auto scalar_objective = [&](latent_parameter_mode_v2 mode,
        const std::vector<float>& z, const std::vector<float>& u,
        const std::vector<float>& a_values, const std::vector<float>& b_values,
        const std::vector<float>& bias_values) {
        std::vector<double> y(z.begin(), z.end());
        for (std::uint32_t c = 0; c < mode_shape.cells; ++c)
            for (auto row = mode_offsets[c]; row < mode_offsets[c + 1]; ++row) {
                const auto m = mode_affected[row];
                const std::size_t block = mode == latent_parameter_mode_v2::shared ? 0 : m;
                for (std::uint32_t o = 0; o < mode_shape.state_width; ++o) {
                    double v = bias_values[block * 2 + o];
                    for (std::uint32_t i = 0; i < mode_shape.state_width; ++i)
                        v += static_cast<double>(a_values[block * 4 + o * 2 + i])
                            * z[(c * 3 + m) * 2 + i];
                    for (std::uint32_t i = 0; i < mode_shape.input_width; ++i)
                        v += static_cast<double>(b_values[block * 4 + o * 2 + i])
                            * u[c * 2 + i];
                    y[(c * 3 + m) * 2 + o] = std::tanh(v);
                }
            }
        double result = 0;
        for (std::size_t i = 0; i < y.size(); ++i) result += y[i] * mode_upstream[i];
        return result;
    };
    const auto check_finite_difference = [](const std::vector<float>& values,
        const std::vector<float>& analytic, auto perturb) {
        constexpr float epsilon = 1.0e-3f;
        assert(values.size() == analytic.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            auto plus = values, minus = values;
            plus[i] += epsilon;
            minus[i] -= epsilon;
            const double expected = (perturb(plus) - perturb(minus)) / (2.0 * epsilon);
            assert(std::abs(static_cast<double>(analytic[i]) - expected) < 1.5e-3);
        }
    };
    const auto validate_mode = [&](latent_parameter_mode_v2 mode) {
        const std::size_t blocks = mode == latent_parameter_mode_v2::shared ? 1 : 3;
        const std::vector<float> a_values(mode_a.begin(), mode_a.begin() + blocks * 4);
        const std::vector<float> b_values(mode_b.begin(), mode_b.begin() + blocks * 4);
        const std::vector<float> bias_values(mode_bias.begin(), mode_bias.begin() + blocks * 2);
        std::vector<float> mode_next(mode_state.size()), zg(mode_state.size()),
            ug(mode_input.size()), ag(a_values.size()), bg(b_values.size()),
            biasg(bias_values.size());
        assert(latent_transition_f32_reference_v2(mode_shape, mode, mode_offsets,
            mode_affected, mode_state, mode_input, a_values, b_values,
            bias_values, mode_next) == status_v1::success);
        const double objective = scalar_objective(mode, mode_state, mode_input,
            a_values, b_values, bias_values);
        double actual_objective = 0;
        for (std::size_t i = 0; i < mode_next.size(); ++i)
            actual_objective += static_cast<double>(mode_next[i]) * mode_upstream[i];
        assert(std::abs(actual_objective - objective) < 2.0e-7);
        assert(latent_transition_vjp_f32_reference_v2(mode_shape, mode,
            mode_offsets, mode_affected, mode_state, mode_input, a_values,
            b_values, bias_values, mode_next, mode_upstream, zg, ug, ag, bg,
            biasg) == status_v1::success);
        check_finite_difference(mode_state, zg, [&](const auto& values) {
            return scalar_objective(mode, values, mode_input, a_values, b_values, bias_values);
        });
        check_finite_difference(mode_input, ug, [&](const auto& values) {
            return scalar_objective(mode, mode_state, values, a_values, b_values, bias_values);
        });
        check_finite_difference(a_values, ag, [&](const auto& values) {
            return scalar_objective(mode, mode_state, mode_input, values, b_values, bias_values);
        });
        check_finite_difference(b_values, bg, [&](const auto& values) {
            return scalar_objective(mode, mode_state, mode_input, a_values, values, bias_values);
        });
        check_finite_difference(bias_values, biasg, [&](const auto& values) {
            return scalar_objective(mode, mode_state, mode_input, a_values, b_values, values);
        });
        // Cell 0/mechanism 1 and cell 1/mechanism 0 are untouched.
        assert(zg[2] == mode_upstream[2] && zg[3] == mode_upstream[3]);
        assert(zg[6] == mode_upstream[6] && zg[7] == mode_upstream[7]);
        return true;
    };
    assert(validate_mode(latent_parameter_mode_v2::shared));
    assert(validate_mode(latent_parameter_mode_v2::per_mechanism));
    std::vector<float> invalid_next(mode_state.size());
    assert(latent_transition_f32_reference_v2(mode_shape,
        static_cast<latent_parameter_mode_v2>(99), mode_offsets, mode_affected,
        mode_state, mode_input, mode_a, mode_b, mode_bias, invalid_next)
        == status_v1::invalid_argument);
    assert(latent_transition_f32_reference_v2(mode_shape,
        latent_parameter_mode_v2::shared, mode_offsets, mode_affected,
        mode_state, mode_input, mode_a, mode_b, mode_bias, invalid_next)
        == status_v1::invalid_argument);
    return 0;
}
