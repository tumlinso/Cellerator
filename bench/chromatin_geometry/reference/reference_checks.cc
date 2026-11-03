#include "reference.hh"
#include "teacher_reference.hh"
#include "../workloads.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cgs {
namespace {

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void close(float actual, float expected, float tolerance,
           const std::string& message) {
    const float scale = std::max(1.0f, std::abs(expected));
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance * scale)
        throw std::runtime_error(message + " expected=" + std::to_string(expected) +
                                 " actual=" + std::to_string(actual));
}

std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void test_half_conversion() {
    using cgs::reference::float_to_half_bits;
    using cgs::reference::half_bits_to_float;
    const std::array<float, 12> values{{
        0.0f, -0.0f, 1.0f, -2.0f, 0x1p-14f, 0x1p-24f,
        65504.0f, 65520.0f, 0.333251953125f, 0.3333740234375f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()}};
    const std::array<std::uint16_t, 12> expected{{
        0x0000u, 0x8000u, 0x3c00u, 0xc000u, 0x0400u, 0x0001u,
        0x7bffu, 0x7c00u, 0x3555u, 0x3556u, 0x7c00u, 0xfc00u}};
    for (std::size_t i = 0; i < values.size(); ++i) {
        check(float_to_half_bits(values[i]) == expected[i],
              "FP16 ties-to-even conversion mismatch at " + std::to_string(i));
        const auto decoded = half_bits_to_float(expected[i]);
        if (std::isinf(values[i])) check(std::isinf(decoded), "FP16 infinity did not round trip");
        else check(bits(decoded) == bits(half_bits_to_float(float_to_half_bits(values[i]))),
                   "FP16 conversion and decode disagree");
    }
    check(float_to_half_bits(std::numeric_limits<float>::quiet_NaN()) & 0x7c00u,
          "FP16 NaN lost its exponent");
}

void test_product_reference() {
    using namespace cgs;
    using cgs::reference::float_to_half_bits;
    const std::array<cellerator::ops::mechanism::GatedProductGroup, 3> groups{{
        {0, 3, 0, 2, 0, 0, 11},
        {3, 2, 2, 2, 0, 1, 12},
        {5, 1, 4, 1, 1, 2, 13}}};
    const std::array<std::uint64_t, 6> args{{0, 1, 1, 1, 2, 3}};
    const std::array<cellerator::ops::mechanism::OutputIncidence, 5> outputs{{
        {0, 1.0f}, {1, -0.5f}, {0, 0.25f}, {2, 2.0f}, {3, 1.0f}}};
    const std::array<std::uint16_t, 8> x{{
        float_to_half_bits(1.0f), float_to_half_bits(2.0f),
        float_to_half_bits(0.0f), float_to_half_bits(3.0f),
        float_to_half_bits(1.5f), float_to_half_bits(2.0f),
        float_to_half_bits(0.0f), float_to_half_bits(4.0f)}};
    const std::array<float, 2> theta{{0.5f, 2.0f}};
    const std::array<std::uint8_t, 6> admitted{{1, 1, 0, 1, 1, 0}};
    const std::array<float, 6> probabilities{{0.8f, 0.6f, 0.2f, 0.7f, 0.9f, 0.4f}};
    const std::array<float, 8> dy{{1.0f, -0.5f, 0.25f, 0.75f,
                                  -1.0f, 0.5f, 1.5f, 2.0f}};
    ProductTopology topology{groups.data(), groups.size(), args.data(), args.size(),
                             outputs.data(), outputs.size(), 4, 3, 2};
    ProblemView problem{};
    problem.topology = topology;
    problem.batch = 2;
    problem.dtype = DType::fp16;
    problem.x = x.data();
    problem.theta = theta.data();
    GateView gate{};
    gate.profile = GateProfile::supplied;
    gate.admitted = admitted.data();
    const auto forward = cgs::reference::product_forward(problem, gate);
    check(std::all_of(forward.y.begin(), forward.y.end(),
                      [](float value) { return std::isfinite(value); }),
          "hard-excluded poison fixture unexpectedly produced nonfinite output");
    close(forward.y[0], 0.5f * (1.0f * 1.0f * 2.0f * 2.0f) +
          0.5f * (1.0f * 2.0f * 0.0f), 1.0e-6f,
          "ordered product overlapping output");

    auto gradients = cgs::reference::product_backward(problem, gate, dy.data());
    check(std::abs(gradients.dx[6]) > 0.0f,
          "zero-valued participant lost its nonzero derivative");
    check(std::abs(gradients.dx[5]) > 0.0f,
          "repeated argument occurrences did not accumulate at one coordinate");
    check(gradients.dtheta[1] == 0.0f,
          "hard-excluded parameter received a numerical gradient");
    check(std::all_of(gradients.dq.begin(), gradients.dq.end(),
                      [](float value) { return value == 0.0f; }) &&
          std::all_of(gradients.dscore.begin(), gradients.dscore.end(),
                      [](float value) { return value == 0.0f; }),
          "supplied Boolean gate unexpectedly received a task gradient");
    auto hard_teacher = gate;
    hard_teacher.profile = GateProfile::teacher;
    const auto teacher_gradients = cgs::reference::product_backward(
        problem, hard_teacher, dy.data());
    check(std::all_of(teacher_gradients.dq.begin(), teacher_gradients.dq.end(),
                      [](float value) { return value == 0.0f; }),
          "hardened teacher gate unexpectedly received a task gradient");

    // Excluded-forward data must not be read by the oracle. NaN in the only
    // participant and parameter for condition 2 remains unobserved.
    auto poison_x = x;
    poison_x[3] = float_to_half_bits(std::numeric_limits<float>::quiet_NaN());
    auto poison_theta = theta;
    poison_theta[1] = std::numeric_limits<float>::quiet_NaN();
    problem.x = poison_x.data();
    problem.theta = poison_theta.data();
    const auto poison_forward = cgs::reference::product_forward(problem, gate);
    check(std::all_of(poison_forward.y.begin(), poison_forward.y.end(),
                      [](float value) { return std::isfinite(value); }),
          "hard-excluded forward read poisoned inputs");

    // STE backward visits the complete derivative set even where primal q=0.
    problem.x = x.data();
    problem.theta = theta.data();
    GateView ste{};
    ste.profile = GateProfile::ste;
    ste.admitted = admitted.data();
    ste.probabilities = probabilities.data();
    ste.revisions.temperature_bits = bits(2.0f);
    const std::array<std::uint32_t, 3> derivative_words{{0x3u, 0x3u, 0x3u}};
    ste.derivative_words = derivative_words.data();
    ste.words_per_condition = 1;
    auto ste_grad = cgs::reference::product_backward(problem, ste, dy.data());
    close(ste_grad.dscore[2], 0.75f * 2.0f * 3.0f * 0.2f * 0.8f / 2.0f,
          1.0e-6f, "dormant STE analytic derivative row 0");
    close(ste_grad.dscore[5], 2.0f * 2.0f * 4.0f * 0.4f * 0.6f / 2.0f,
          1.0e-6f, "dormant STE analytic derivative row 1");
    check(ste_grad.dq[2] != 0.0f && ste_grad.dq[5] != 0.0f,
          "STE omitted a dormant derivative-candidate dQ");
    check(ste_grad.dtheta[1] == 0.0f,
          "dormant STE candidate changed the numerical parameter gradient");
    check(ste_grad.dx[3] == 0.0f && ste_grad.dx[7] == 0.0f,
          "dormant STE candidate changed participant gradients");

    // Empty batches are valid and do not dereference input or admission data.
    problem.batch = 0;
    problem.x = nullptr;
    GateView empty_gate{};
    empty_gate.profile = GateProfile::supplied;
    check(cgs::reference::product_forward(problem, empty_gate).y.empty(),
          "empty batch did not produce an empty output");
}

void test_product_arity_boundaries() {
    using namespace cgs;
    for (std::uint64_t arity : {32ull, 33ull}) {
        std::vector<cellerator::ops::mechanism::GatedProductGroup> groups{
            {0, arity, 0, 1, 0, 0, 101}};
        std::vector<std::uint64_t> arguments(static_cast<std::size_t>(arity));
        std::vector<std::uint16_t> x(static_cast<std::size_t>(arity));
        for (std::uint64_t i = 0; i < arity; ++i) {
            arguments[i] = i;
            x[i] = cgs::reference::float_to_half_bits(1.0f);
        }
        const std::array<cellerator::ops::mechanism::OutputIncidence, 1> outputs{{{0, 1}}};
        const std::array<float, 1> theta{{1.0f}};
        const std::array<std::uint8_t, 1> admission{{1}};
        ProductTopology topology{groups.data(), groups.size(), arguments.data(), arguments.size(),
                                 outputs.data(), outputs.size(), arity, 1, 1};
        ProblemView problem{};
        problem.topology = topology;
        problem.batch = 1;
        problem.dtype = DType::fp16;
        problem.x = x.data();
        problem.theta = theta.data();
        GateView gate{};
        gate.profile = GateProfile::supplied;
        gate.admitted = admission.data();
        const auto forward = cgs::reference::product_forward(problem, gate);
        check(forward.y.size() == arity && forward.y.front() == 1.0f,
              "product arity boundary forward mismatch");
        std::vector<float> dy(static_cast<std::size_t>(arity), 0.0f);
        dy[0] = 1.0f;
        const auto backward = cgs::reference::product_backward(problem, gate, dy.data());
        check(backward.dx.front() == 1.0f,
              "product arity boundary derivative mismatch");
    }
    const std::array<cellerator::ops::mechanism::GatedProductGroup, 1> empty_group{{
        {0, 0, 0, 1, 0, 0, 100}}};
    const std::array<cellerator::ops::mechanism::OutputIncidence, 1> empty_outputs{{{0, 1.0f}}};
    const std::array<float, 1> empty_theta{{2.0f}};
    const std::array<std::uint8_t, 1> empty_admission{{1}};
    ProductTopology empty_topology{empty_group.data(), empty_group.size(), nullptr, 0,
                                   empty_outputs.data(), empty_outputs.size(), 1, 1, 1};
    ProblemView empty_problem{};
    empty_problem.topology = empty_topology;
    empty_problem.batch = 1;
    empty_problem.dtype = DType::fp16;
    empty_problem.x = nullptr;
    empty_problem.theta = empty_theta.data();
    GateView empty_gate{};
    empty_gate.profile = GateProfile::supplied;
    empty_gate.admitted = empty_admission.data();
    check(cgs::reference::product_forward(empty_problem, empty_gate).y[0] == 2.0f,
          "empty participant product should be the multiplicative identity");
}

void test_grouped_linear_reference() {
    using namespace cgs;
    using cgs::reference::float_to_half_bits;
    const std::array<cellerator::ops::mechanism::GatedProductGroup, 2> groups{{
        {0, 2, 0, 2, 0, 0, 51},
        {2, 2, 2, 2, 0, 0, 52}}};
    const std::array<std::uint64_t, 4> args{{0, 1, 1, 2}};
    const std::array<cellerator::ops::mechanism::OutputIncidence, 4> outputs{{
        {0, 1.0f}, {1, 0.5f}, {1, 1.0f}, {2, 1.0f}}};
    const std::array<std::uint16_t, 8> x16{{
        float_to_half_bits(1.25f), float_to_half_bits(0.75f),
        float_to_half_bits(-1.0f), float_to_half_bits(2.5f),
        float_to_half_bits(-0.5f), float_to_half_bits(1.5f),
        float_to_half_bits(2.0f), float_to_half_bits(0.25f)}};
    const std::array<std::uint16_t, 4> weights16{{
        float_to_half_bits(0.333251953125f), float_to_half_bits(-0.2f),
        float_to_half_bits(0.1f), float_to_half_bits(0.4f)}};
    const std::array<std::uint64_t, 1> weight_offsets{{0}};
    const std::array<std::uint64_t, 3> local_weight_offsets{{0, 4, 8}};
    const std::array<std::uint8_t, 2> admission{{1, 1}};
    ProductTopology topology{groups.data(), groups.size(), args.data(), args.size(),
                             outputs.data(), outputs.size(), 4, 1, 1};
    ProblemView problem{};
    problem.topology = topology;
    problem.batch = 2;
    problem.dtype = DType::fp16;
    problem.x = x16.data();
    problem.weights = weights16.data();
    problem.weight_offsets = weight_offsets.data();
    problem.group_weight_offsets = local_weight_offsets.data();
    problem.weight_count = weights16.size();
    problem.group_weight_count = 8;
    GateView gate{};
    gate.profile = GateProfile::supplied;
    gate.admitted = admission.data();
    const auto forward = cgs::reference::grouped_linear_forward(problem, gate);
    check(forward.group_f.size() == 8, "linear local F extent mismatch");
    for (float value : forward.group_f)
        check(cgs::reference::round_value(value, DType::fp16) == value,
              "linear local F did not round to FP16 storage");
    const std::array<float, 8> dy{{1.0f, -0.5f, 0.25f, 1.5f,
                                  -0.75f, 0.5f, 2.0f, -1.0f}};
    const auto backward = cgs::reference::grouped_linear_backward(
        problem, gate, dy.data(), forward);
    const auto validation_bounds = cgs::reference::grouped_linear_validation_bounds(
        problem, gate, dy.data(), forward, backward);
    check(backward.group_df.size() == 8 && backward.group_dx.size() == 8 &&
          backward.group_dw.size() == 8,
          "linear staged gradient extents mismatch");
    check(backward.dweights.size() == 4,
          "shared linear parameter extent mismatch");
    check(std::all_of(backward.dq.begin(), backward.dq.end(),
                      [](float value) { return value == 0.0f; }) &&
          std::all_of(backward.dscore.begin(), backward.dscore.end(),
                      [](float value) { return value == 0.0f; }),
          "supplied grouped-linear gate unexpectedly received a task gradient");
    check(validation_bounds.y.size() == forward.y.size() &&
          validation_bounds.dx.size() == backward.dx.size() &&
          validation_bounds.dweights.size() == backward.dweights.size() &&
          validation_bounds.dq.size() == backward.dq.size() &&
          validation_bounds.group_dx.size() == backward.group_dx.size() &&
          validation_bounds.group_dw.size() == backward.group_dw.size(),
          "compound linear bounds have wrong extents");
    for (std::size_t i = 0; i < forward.y.size(); ++i)
        check(cgs::reference::within_absolute_bound(
                  forward.y[i], forward.y[i], validation_bounds.y[i]),
              "reference linear output must satisfy its own compound bound");
    check(std::all_of(validation_bounds.dq.begin(), validation_bounds.dq.end(),
                      [](float value) { return value == 0.0f; }),
          "hard grouped-linear dq validation bounds must be zero");
    check(cgs::reference::grouped_linear_allowed_error(
              DType::fp16, 1.0f, 4, 2.0) > 0.0f,
          "frozen GroupedLinear compound bound omitted FP16 rounding");
    for (std::size_t i = 0; i < backward.dweights.size(); ++i)
        close(backward.dweights[i], backward.group_dw[i] + backward.group_dw[4 + i],
              1.0e-6f, "shared local dW canonical reduction");
    for (std::size_t i = 0; i < backward.group_dw.size(); ++i)
        check(cgs::reference::round_value(backward.group_dw[i], DType::fp16) ==
                  backward.group_dw[i], "local dW did not round to FP16");

    // FP32 is an independent validation arm with no staging conversion.
    const std::array<float, 8> x32{{1.25f, 0.75f, -1.0f, 2.5f,
                                    -0.5f, 1.5f, 2.0f, 0.25f}};
    const std::array<float, 4> weights32{{0.333251953125f, -0.2f, 0.1f, 0.4f}};
    problem.dtype = DType::fp32;
    problem.x = x32.data();
    problem.weights = weights32.data();
    const auto fp32 = cgs::reference::grouped_linear_forward(problem, gate);
    for (std::size_t i = 0; i < fp32.group_f.size(); ++i)
        check(fp32.group_f[i] == cgs::reference::round_value(
                  fp32.group_f[i], DType::fp32), "FP32 linear changed local output");
    const auto fp32_backward = cgs::reference::grouped_linear_backward(
        problem, gate, dy.data(), fp32);
    check(fp32_backward.dweights.size() == weights32.size(),
          "FP32 grouped linear reference extent mismatch");
    GateView teacher_gate = gate;
    teacher_gate.profile = GateProfile::teacher;
    const auto teacher_backward = cgs::reference::grouped_linear_backward(
        problem, teacher_gate, dy.data(), fp32);
    check(std::all_of(teacher_backward.dq.begin(), teacher_backward.dq.end(),
                      [](float value) { return value == 0.0f; }),
          "hardened teacher grouped-linear gate unexpectedly received a task gradient");

    // FP32 consumer arithmetic is smooth here, so a centered difference
    // independently checks one overlapping participant VJP.
    auto perturbed = x32;
    constexpr float step = 1.0e-3f;
    const auto objective = [&](const std::array<float, 8>& values) {
        problem.x = values.data();
        const auto output = cgs::reference::grouped_linear_forward(problem, gate);
        float total = 0.0f;
        for (std::size_t i = 0; i < output.y.size(); ++i) total += output.y[i] * dy[i];
        return total;
    };
    perturbed[0] += step;
    const float plus = objective(perturbed);
    perturbed[0] -= 2.0f * step;
    const float minus = objective(perturbed);
    close((plus - minus) / (2.0f * step), fp32_backward.dx[0], 2.0e-3f,
          "FP32 grouped linear centered-difference VJP");

    // A dormant STE row has no primal local gradients, while the declared
    // derivative set still reads its inputs to form dscore.
    problem.x = x32.data();
    GateView ste{};
    ste.profile = GateProfile::ste;
    const std::array<std::uint8_t, 2> ste_admission{{1, 0}};
    const std::array<float, 2> ste_probability{{0.75f, 0.25f}};
    const std::array<std::uint32_t, 1> ste_derivative{{0x3u}};
    ste.admitted = ste_admission.data();
    ste.probabilities = ste_probability.data();
    ste.derivative_words = ste_derivative.data();
    ste.words_per_condition = 1;
    ste.revisions.temperature_bits = bits(1.0f);
    const auto ste_forward = cgs::reference::grouped_linear_forward(problem, ste);
    const auto ste_backward = cgs::reference::grouped_linear_backward(
        problem, ste, dy.data(), ste_forward);
    const auto ste_bounds = cgs::reference::grouped_linear_validation_bounds(
        problem, ste, dy.data(), ste_forward, ste_backward);
    check(ste_backward.dscore[1] != 0.0f,
          "dormant grouped-linear STE lost the full derivative candidate");
    check(ste_bounds.dscore[1] > 0.0f,
          "dormant grouped-linear STE lacks an analytic gradient bound");
    check(ste_backward.group_df[4] == 0.0f && ste_backward.group_dx[4] == 0.0f,
          "dormant grouped-linear row acquired primal local gradients");
}

void test_teacher_reference() {
    using namespace cgs::reference;
    const std::array<float, 8> logits{{-20.0f, -2.0f, -0.25f, 0.25f,
                                       0.0f, 2.0f, 20.0f, 1.0f}};
    const std::array<float, 8> labels{{0.0f, 1.0f, 0.0f, 1.0f,
                                       0.0f, 1.0f, 1.0f, 0.25f}};
    const auto table = teacher_reference(logits, labels);
    check(table.decision[4] == 1u && (table.table & (1u << 4u)) != 0u,
          "teacher threshold equality must harden to true");
    for (std::uint8_t row = 0; row < 8; ++row) {
        const bool a = (row & 4u) != 0u;
        const bool b = (row & 2u) != 0u;
        const bool c = (row & 1u) != 0u;
        check(teacher_lookup(table.table, a, b, c) == table.decision[row],
              "teacher LUT row mapping mismatch");
    }
    const std::array<std::uint8_t, 4> rows{{0, 0, 4, 7}};
    const std::array<float, 4> sample_labels{{1.0f, 0.0f, 1.0f, 0.25f}};
    const auto minibatch = teacher_batch_reference(
        logits, rows.data(), sample_labels.data(), rows.size());
    const auto row0 = teacher_example_reference(logits[0], 0.5f);
    close(minibatch.mean_logit_gradient[0], row0.dlogit * 0.5f,
          1.0e-6f, "teacher example-mean gradient scaling");
    const std::array<std::uint64_t, 8> counts{{2, 0, 0, 0, 1, 0, 0, 1}};
    const std::array<double, 8> positives{{1, 0, 0, 0, 1, 0, 0, 0.25}};
    const auto row_means = teacher_row_mean_reference(logits, counts, positives);
    close(row_means.mean_logit_gradient[0], row0.dlogit, 1.0e-6f,
          "teacher independent-row gradient");
    close(row_means.mean_logit_gradient[4],
          teacher_example_reference(logits[4], 1.0f).dlogit, 1.0e-6f,
          "teacher threshold row target gradient");
    check(row_means.mean_logit_gradient[1] == 0.0f &&
          row_means.row_has_examples[1] == 0u,
          "empty teacher row must remain unchanged");
}

void test_product_centered_difference() {
    using namespace cgs;
    const std::array<cellerator::ops::mechanism::GatedProductGroup, 1> groups{{
        {0, 2, 0, 1, 0, 0, 91}}};
    const std::array<std::uint64_t, 2> args{{0, 1}};
    const std::array<cellerator::ops::mechanism::OutputIncidence, 1> outputs{{{0, 1.0f}}};
    const std::array<float, 2> input{{0.7f, 1.3f}};
    const std::array<float, 1> theta{{0.4f}};
    const std::array<std::uint8_t, 1> admission{{1}};
    const std::array<float, 1> dy{{1.0f}};
    ProductTopology topology{groups.data(), groups.size(), args.data(), args.size(),
                             outputs.data(), outputs.size(), 2, 1, 1};
    ProblemView problem{};
    problem.topology = topology;
    problem.batch = 1;
    problem.dtype = DType::fp32;
    problem.x = input.data();
    problem.theta = theta.data();
    GateView gate{};
    gate.profile = GateProfile::supplied;
    gate.admitted = admission.data();
    const auto gradient = cgs::reference::product_backward(problem, gate, dy.data());
    std::array<float, 2> perturbed = input;
    constexpr float step = 1.0e-3f;
    perturbed[0] += step;
    problem.x = perturbed.data();
    const auto plus = cgs::reference::product_forward(problem, gate).y[0];
    perturbed[0] -= 2.0f * step;
    problem.x = perturbed.data();
    const auto minus = cgs::reference::product_forward(problem, gate).y[0];
    close((plus - minus) / (2.0f * step), gradient.dx[0], 1.0e-4f,
          "FP32 product centered-difference VJP");
}

void test_workload_manifest() {
    const auto& manifest = workload_manifest();
    check(manifest.size() >= 10, "workload manifest lost required contrasts");
    bool saw_7 = false, saw_2304 = false;
    bool saw_coherent = false, saw_diverse = false;
    bool saw_sparse = false, saw_dense = false;
    bool saw_low = false, saw_high = false;
    bool saw_frozen = false, saw_update = false;
    bool saw_legacy7_complete = false, saw_legacy2304_complete = false;
    bool saw_64_complete = false;
    bool saw_producer_diagnostic = false;
    bool reuse[3]{false, false, false};
    bool changes[4]{false, false, false, false};
    for (const auto& spec : manifest) {
        saw_7 |= spec.dag_nodes == 7;
        saw_2304 |= spec.dag_nodes == 2304;
        saw_coherent |= spec.program_pattern == ProgramPattern::coherent;
        saw_diverse |= spec.program_pattern == ProgramPattern::diverse;
        saw_sparse |= spec.activity_pattern == ActivityPattern::sparse;
        saw_dense |= spec.activity_pattern == ActivityPattern::dense;
        saw_low |= spec.overlap_pattern == OverlapPattern::low;
        saw_high |= spec.overlap_pattern == OverlapPattern::high;
        saw_frozen |= spec.parameter_pattern == ParameterPattern::frozen;
        saw_update |= spec.parameter_pattern == ParameterPattern::update_every_step;
        saw_producer_diagnostic |=
            spec.role == WorkloadRole::producer_only_diagnostic;
        saw_legacy7_complete |= spec.dag_nodes == 7 && spec.legacy_root_map &&
            spec.role == WorkloadRole::complete_consumer;
        saw_legacy2304_complete |= spec.dag_nodes == 2304 && spec.legacy_root_map &&
            spec.role == WorkloadRole::complete_consumer;
        saw_64_complete |= spec.condition_edges == 64 && spec.groups == 64 &&
            spec.role == WorkloadRole::complete_consumer;
        reuse[spec.reuse_horizon == 1 ? 0 : spec.reuse_horizon == 16 ? 1 : 2] = true;
        changes[spec.predicate_change_percent == 0 ? 0 :
                spec.predicate_change_percent == 1 ? 1 :
                spec.predicate_change_percent == 10 ? 2 : 3] = true;

        const auto fixture = materialize_workload(spec);
        check(fixture.programs.size() == 1,
              "condition edges must share one node bank");
        const auto& dag = fixture.programs.front();
        check(dag.nodes.size() == spec.dag_nodes && dag.root == spec.dag_nodes - 1 &&
              fixture.edge_roots.size() == spec.condition_edges,
              "condition DAG or edge-root identity mismatch");
        check(!fixture.demanded_roots.empty() && !fixture.demanded_nodes.empty() &&
              fixture.demanded_levels.size() == dag.levels.size(),
              "demanded root/node closure metadata is missing");
        check(fixture.predicate_features == (spec.dag_nodes == 7 ? 4u : 128u),
              "predicate input width differs from the historical bank");
        std::vector<unsigned> seen(dag.nodes.size(), 0u);
        for (const auto& level : dag.levels)
            for (auto node : level) {
                check(node < dag.nodes.size(), "condition level node out of bounds");
                ++seen[node];
            }
        check(std::all_of(seen.begin(), seen.end(),
                          [](unsigned value) { return value == 1u; }),
              "condition levels do not partition DAG nodes");
        for (std::size_t i = 0; i < dag.nodes.size(); ++i) {
            const auto& node = dag.nodes[i];
            if (node.op != PredicateOp::feature)
                check(node.child_count != 0 &&
                      node.first_child + node.child_count <= i,
                      "condition child ranges are not topological");
        }
        std::vector<std::uint8_t> expected_roots(dag.nodes.size(), 0u);
        for (const auto root : fixture.edge_roots) expected_roots[root] = 1u;
        std::vector<std::uint8_t> expected_demanded = expected_roots;
        for (std::size_t reverse = dag.nodes.size(); reverse != 0; --reverse) {
            const auto node_id = reverse - 1u;
            if (!expected_demanded[node_id]) continue;
            const auto& node = dag.nodes[node_id];
            if (node.op == PredicateOp::all || node.op == PredicateOp::any) {
                for (std::uint32_t child = node.first_child;
                     child < node.first_child + node.child_count; ++child)
                    expected_demanded[child] = 1u;
            } else if (node.op == PredicateOp::logical_not) {
                expected_demanded[node.first_child] = 1u;
            }
        }
        std::vector<std::uint32_t> expected_root_ids, expected_node_ids;
        for (std::uint32_t node = 0; node < dag.nodes.size(); ++node) {
            if (expected_roots[node]) expected_root_ids.push_back(node);
            if (expected_demanded[node]) expected_node_ids.push_back(node);
        }
        check(fixture.demanded_roots == expected_root_ids &&
              fixture.demanded_nodes == expected_node_ids,
              "unique demanded root or ancestor IDs differ from edge map");
        for (std::size_t level = 0; level < dag.levels.size(); ++level) {
            std::vector<std::uint32_t> expected;
            for (const auto node : dag.levels[level])
                if (expected_demanded[node]) expected.push_back(node);
            check(fixture.demanded_levels[level] == expected,
                  "demanded node levels differ from exact root closure");
        }
        if (spec.dag_nodes == 7) {
            for (std::uint32_t feature = 0; feature < 4; ++feature)
                check(dag.nodes[feature].op == PredicateOp::feature &&
                      dag.nodes[feature].feature == feature,
                      "historical seven-node predicate leaf changed");
            check(dag.nodes[4].op == PredicateOp::all &&
                  dag.nodes[4].first_child == 0 && dag.nodes[4].child_count == 2 &&
                  dag.nodes[5].op == PredicateOp::any &&
                  dag.nodes[5].first_child == 2 && dag.nodes[5].child_count == 2 &&
                  dag.nodes[6].op == PredicateOp::all &&
                  dag.nodes[6].first_child == 0 && dag.nodes[6].child_count == 3,
                  "historical seven-node operator topology changed");
            if (spec.legacy_root_map) {
                const auto prefix = std::max(1u, (spec.condition_edges + 15u) / 16u);
                constexpr std::uint32_t other_roots[]{1u, 2u, 3u, 5u};
                for (std::uint32_t edge = 0; edge < spec.condition_edges; ++edge) {
                    const auto expected = edge < prefix
                        ? (edge % 3u == 0u ? 0u : edge % 3u == 1u ? 4u : 6u)
                        : other_roots[(edge - prefix) % 4u];
                    check(fixture.edge_roots[edge] == expected,
                          "historical seven-node root map changed");
                }
            }
        } else {
            check(dag.roots.size() == 2176u,
                  "2,304-node bank lost its 2,176 distinct roots");
            for (std::uint32_t leaf = 0; leaf < 128; ++leaf)
                check(dag.nodes[leaf].op == PredicateOp::feature &&
                      dag.nodes[leaf].feature == leaf,
                      "historical 2,304-node predicate leaf changed");
            std::uint32_t start = 0, width = 2;
            for (std::uint32_t root_index = 0; root_index < 2176; ++root_index) {
                const auto root = 128u + root_index;
                check(dag.roots[root_index] == root &&
                      dag.nodes[root].op == PredicateOp::all &&
                      dag.nodes[root].first_child == start &&
                      dag.nodes[root].child_count == width,
                      "historical contiguous-window root topology changed");
                if (spec.legacy_root_map)
                    check(fixture.edge_roots[root_index] == root,
                          "historical 2,304-node edge map changed");
                ++start;
                if (start + width > 128u) {
                    ++width;
                    start = 0;
                }
            }
        }

        const std::uint64_t state_count =
            static_cast<std::uint64_t>(spec.cells) * fixture.predicate_features;
        const std::uint64_t expected_changes =
            (state_count * spec.predicate_change_percent + 50u) / 100u;
        std::uint64_t observed_changes = 0;
        for (std::uint32_t cell = 0; cell < spec.cells; ++cell)
            for (std::uint32_t feature = 0; feature < spec.features; ++feature) {
                const auto at = static_cast<std::size_t>(cell) * spec.features + feature;
                const bool changed = fixture.accessibility[at] !=
                                     fixture.accessibility_after[at];
                observed_changes += changed;
                if (feature >= fixture.predicate_features)
                    check(!changed, "predicate update changed an undeclared input");
            }
        check(observed_changes == expected_changes,
              "input predicate change count differs from frozen percentage");
        check(fixture.changed_input_features.size() == expected_changes,
              "changed input feature map lost changes");
        check(fixture.changed_input_offsets.size() == spec.cells + 1u &&
              fixture.dirty_node_offsets.size() == spec.cells + 1u &&
              fixture.predicates_before.size() ==
                  static_cast<std::size_t>(spec.cells) * spec.condition_edges &&
              fixture.predicates_after.size() == fixture.predicates_before.size(),
              "condition delta offsets have wrong extents");
        std::uint64_t root_changes = 0;
        for (std::uint32_t cell = 0; cell < spec.cells; ++cell) {
            const auto* before = fixture.accessibility.data() +
                                 static_cast<std::size_t>(cell) * spec.features;
            const auto* after = fixture.accessibility_after.data() +
                                static_cast<std::size_t>(cell) * spec.features;
            const auto before_nodes = evaluate_condition_nodes(dag, before, spec.features);
            const auto after_nodes = evaluate_condition_nodes(dag, after, spec.features);
            for (std::uint32_t edge = 0; edge < spec.condition_edges; ++edge) {
                const auto at = static_cast<std::size_t>(cell) * spec.condition_edges + edge;
                const auto root = fixture.edge_roots[edge];
                check(fixture.predicates_before[at] == before_nodes[root] &&
                      fixture.predicates_after[at] == after_nodes[root],
                      "recorded condition root does not match DAG evaluation");
                root_changes += before_nodes[root] != after_nodes[root];
            }
            const auto begin = fixture.dirty_node_offsets[cell];
            const auto end = fixture.dirty_node_offsets[cell + 1u];
            std::vector<std::uint8_t> observed_dirty(dag.nodes.size(), 0u);
            for (auto cursor = begin; cursor < end; ++cursor) {
                const auto node = fixture.dirty_nodes[cursor];
                check(node < dag.nodes.size() &&
                      (cursor == begin || fixture.dirty_nodes[cursor - 1u] < node),
                      "dirty closure IDs are invalid or unsorted");
                observed_dirty[node] = 1u;
            }
            const auto changed_begin = fixture.changed_input_offsets[cell];
            const auto changed_end = fixture.changed_input_offsets[cell + 1u];
            for (std::size_t node_index = 0; node_index < dag.nodes.size(); ++node_index) {
                const auto& node = dag.nodes[node_index];
                bool expected_dirty = false;
                if (node.op == PredicateOp::feature) {
                    expected_dirty = std::find(
                        fixture.changed_input_features.begin() + changed_begin,
                        fixture.changed_input_features.begin() + changed_end,
                        static_cast<std::uint32_t>(node.feature)) !=
                        fixture.changed_input_features.begin() + changed_end;
                } else if (node.op == PredicateOp::logical_not) {
                    expected_dirty = observed_dirty[node.first_child] != 0u;
                } else {
                    for (std::uint32_t child = node.first_child;
                         child < node.first_child + node.child_count; ++child)
                        expected_dirty |= observed_dirty[child] != 0u;
                }
                check((observed_dirty[node_index] != 0u) == expected_dirty,
                      "dirty-node list differs from exact shared-bank dependency closure");
            }
        }
        check(root_changes == fixture.root_decision_changes,
              "recorded root decision delta count is inconsistent");
        check(fixture.distal_memberships ==
                  static_cast<std::uint64_t>(spec.groups) * 4u,
              "explicit distal regulatory memberships missing");
        check(fixture.group_program.size() == spec.groups,
              "numerical group-to-edge map has the wrong extent");
        if (spec.role == WorkloadRole::complete_consumer) {
            check(spec.groups == spec.condition_edges,
                  "complete consumer case does not cover every root");
            for (std::uint32_t group = 0; group < spec.groups; ++group)
                check(fixture.group_program[group] == group,
                      "complete consumer group does not bind one-to-one to an edge");
            if (spec.condition_edges == 64u) {
                const auto expected_unique = spec.program_pattern ==
                    ProgramPattern::coherent ? 1u : 64u;
                check(fixture.demanded_roots.size() == expected_unique,
                      "64-edge complete case has an unexpected selected-root count");
                if (spec.dag_nodes == 2304u)
                    check(fixture.demanded_nodes.size() < dag.nodes.size(),
                          "64-root large-DAG control did not select a smaller closure");
            } else {
                const auto expected_unique = spec.dag_nodes == 7u
                    ? 7u : spec.condition_edges;
                check(fixture.demanded_roots.size() == expected_unique,
                      "legacy complete case lost a distinct demanded root");
            }
        } else {
            check(spec.groups == 64 && spec.groups < spec.condition_edges,
                  "producer-only case must retain its labeled 64-group diagnostic scope");
        }
        for (std::uint64_t i = 0; i < fixture.group_features.size(); ++i) {
            const bool distal_slot = (i % 8u) >= 4u;
            check((fixture.group_features[i] >= spec.features / 2u) == distal_slot,
                  "distal membership fixture does not preserve explicit IDs");
        }
    }
    check(saw_7 && saw_2304 && saw_coherent && saw_diverse && saw_sparse && saw_dense &&
          saw_low && saw_high && saw_frozen && saw_update &&
          saw_legacy7_complete && saw_legacy2304_complete && saw_producer_diagnostic &&
          saw_64_complete &&
          reuse[0] && reuse[1] && reuse[2] &&
          changes[0] && changes[1] && changes[2] && changes[3],
          "frozen workload manifest misses a required regime");
}

} // namespace

bool run_reference_checks(std::string* error) {
    try {
        test_half_conversion();
        test_product_reference();
        test_product_arity_boundaries();
        test_product_centered_difference();
        test_grouped_linear_reference();
        test_teacher_reference();
        test_workload_manifest();
        if (error != nullptr) error->clear();
        return true;
    } catch (const std::exception& exception) {
        if (error != nullptr) *error = exception.what();
        return false;
    }
}

} // namespace cgs
