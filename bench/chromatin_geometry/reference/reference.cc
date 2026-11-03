#include "reference.hh"
#include "teacher_reference.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace cgs::reference {
namespace {

std::uint64_t group_output_total(const ProblemView& problem) {
    std::uint64_t total = 0;
    for (std::uint64_t g = 0; g < problem.topology.group_count; ++g) {
        const auto& group = problem.topology.groups[g];
        if (group.output_count > std::numeric_limits<std::uint64_t>::max() - total)
            throw std::overflow_error("group output extent overflows");
        total += group.output_count;
    }
    return total;
}

bool gate_bit(const GateView& gate, std::uint64_t batch,
              std::uint64_t condition, std::uint64_t conditions) {
    if (gate.admitted != nullptr)
        return gate.admitted[batch * conditions + condition] != 0;
    if (gate.primal_words != nullptr) {
        const std::uint64_t words = gate.words_per_condition;
        return ((gate.primal_words[condition * words + batch / 32] >>
                 (batch & 31u)) & 1u) != 0;
    }
    throw std::invalid_argument("reference gate has no primal admission data");
}

bool derivative_bit(const GateView& gate, std::uint64_t batch,
                    std::uint64_t condition, std::uint64_t conditions) {
    if (gate.derivative_words != nullptr) {
        const std::uint64_t words = gate.words_per_condition;
        return ((gate.derivative_words[condition * words + batch / 32] >>
                 (batch & 31u)) & 1u) != 0;
    }
    // A missing derivative mask means the conservative complete set. In
    // particular, it never treats the hard forward mask as derivative support.
    (void) conditions;
    return true;
}

float condition_probability(const GateView& gate, std::uint64_t batch,
                            std::uint64_t condition,
                            std::uint64_t conditions) {
    const bool admitted = gate_bit(gate, batch, condition, conditions);
    if (gate.profile == GateProfile::ste) {
        if (gate.probabilities == nullptr)
            throw std::invalid_argument("STE reference needs probabilities");
        return admitted
            ? gate.probabilities[batch * conditions + condition]
            : 0.0f;
    }
    return admitted ? 1.0f : 0.0f;
}

bool primal_active(const GateView& gate, std::uint64_t batch,
                   std::uint64_t condition, std::uint64_t conditions) {
    return gate_bit(gate, batch, condition, conditions);
}

bool backward_active(const GateView& gate, std::uint64_t batch,
                     std::uint64_t condition, std::uint64_t conditions) {
    if (gate.profile != GateProfile::ste)
        return primal_active(gate, batch, condition, conditions);
    return derivative_bit(gate, batch, condition, conditions);
}

float product_for(const ProblemView& problem, std::uint64_t batch,
                  const cellerator::ops::mechanism::GatedProductGroup& group) {
    float product = 1.0f;
    for (std::uint64_t slot = 0; slot < group.argument_count; ++slot) {
        const auto coordinate = problem.topology.arguments[group.argument_begin + slot];
        product *= read_value(problem.x, problem.dtype,
                              batch * problem.topology.coordinates + coordinate);
    }
    return product;
}

float fp16_ulp(float value) noexcept {
    const float magnitude = std::abs(value);
    if (magnitude < 0x1p-14f) return 0x1p-24f;
    return std::ldexp(1.0f, std::ilogb(magnitude) - 10);
}

void require_product(const ProblemView& problem) {
    if (problem.topology.group_count != 0 && problem.topology.groups == nullptr)
        throw std::invalid_argument("product reference has no groups");
    if (problem.topology.argument_count != 0 && problem.topology.arguments == nullptr)
        throw std::invalid_argument("product reference has no argument slots");
    if (problem.topology.output_count != 0 && problem.topology.outputs == nullptr)
        throw std::invalid_argument("product reference has no output incidences");
    bool needs_x = problem.include_baseline && problem.topology.coordinates != 0;
    for (std::uint64_t g = 0; g < problem.topology.group_count; ++g)
        needs_x = needs_x || problem.topology.groups[g].argument_count != 0;
    if (problem.batch != 0 && needs_x && problem.x == nullptr)
        throw std::invalid_argument("product reference has no required input values");
    if (problem.topology.parameters != 0 && problem.theta == nullptr)
        throw std::invalid_argument("product reference has no theta values");
}

void require_linear(const ProblemView& problem) {
    if (problem.topology.group_count != 0 && problem.topology.groups == nullptr)
        throw std::invalid_argument("linear reference has no groups");
    if (problem.topology.argument_count != 0 && problem.topology.arguments == nullptr)
        throw std::invalid_argument("linear reference has no argument slots");
    if (problem.topology.output_count != 0 && problem.topology.outputs == nullptr)
        throw std::invalid_argument("linear reference has no output incidences");
    if (problem.batch != 0 && problem.topology.coordinates != 0 && problem.x == nullptr)
        throw std::invalid_argument("linear reference has no input values");
    if (problem.weight_count != 0 && problem.weights == nullptr)
        throw std::invalid_argument("linear reference has no weights");
    if (problem.topology.group_count != 0 && problem.group_weight_offsets == nullptr)
        throw std::invalid_argument("linear reference has no group weight offsets");
    if (problem.topology.parameters != 0 && problem.weight_offsets == nullptr)
        throw std::invalid_argument("linear reference has no parameter weight offsets");
}

float baseline_value(const ProblemView& problem, std::uint64_t batch,
                    std::uint64_t coordinate) {
    return problem.include_baseline
        ? problem.beta * read_value(problem.x, problem.dtype,
                                    batch * problem.topology.coordinates + coordinate)
        : 0.0f;
}

} // namespace

std::uint16_t float_to_half_bits(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exponent = (bits >> 23) & 0xffu;
    const std::uint32_t mantissa = bits & 0x7fffffu;

    if (exponent == 0xffu) {
        if (mantissa == 0u) return static_cast<std::uint16_t>(sign | 0x7c00u);
        std::uint32_t payload = mantissa >> 13;
        if (payload == 0u) payload = 1u;
        return static_cast<std::uint16_t>(sign | 0x7c00u | payload);
    }

    const int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31)
        return static_cast<std::uint16_t>(sign | 0x7c00u);

    if (half_exponent <= 0) {
        if (half_exponent < -10)
            return static_cast<std::uint16_t>(sign);
        const std::uint32_t significand = mantissa | 0x800000u;
        const unsigned shift = static_cast<unsigned>(14 - half_exponent);
        std::uint32_t rounded = significand >> shift;
        const std::uint32_t remainder_mask = (std::uint32_t{1} << shift) - 1u;
        const std::uint32_t remainder = significand & remainder_mask;
        const std::uint32_t halfway = std::uint32_t{1} << (shift - 1u);
        if (remainder > halfway ||
            (remainder == halfway && (rounded & 1u) != 0u))
            ++rounded;
        return static_cast<std::uint16_t>(sign | rounded);
    }

    std::uint32_t rounded_mantissa = mantissa >> 13;
    const std::uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u ||
        (remainder == 0x1000u && (rounded_mantissa & 1u) != 0u)) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400u) {
            const std::uint32_t next_exp = static_cast<std::uint32_t>(half_exponent + 1);
            if (next_exp >= 31u)
                return static_cast<std::uint16_t>(sign | 0x7c00u);
            return static_cast<std::uint16_t>(sign | (next_exp << 10));
        }
    }
    return static_cast<std::uint16_t>(sign |
        (static_cast<std::uint32_t>(half_exponent) << 10) | rounded_mantissa);
}

float half_bits_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    std::uint32_t exponent = (bits >> 10) & 0x1fu;
    std::uint32_t mantissa = bits & 0x03ffu;
    std::uint32_t out = 0;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            out = sign;
        } else {
            int e = -14;
            while ((mantissa & 0x0400u) == 0u) {
                mantissa <<= 1;
                --e;
            }
            mantissa &= 0x03ffu;
            out = sign | (static_cast<std::uint32_t>(e + 127) << 23) |
                  (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        out = sign | 0x7f800000u | (mantissa << 13);
    } else {
        exponent = exponent - 15u + 127u;
        out = sign | (exponent << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

float read_value(const void* values, DType dtype, std::uint64_t index) {
    if (values == nullptr) throw std::invalid_argument("null numerical input");
    if (dtype == DType::fp16)
        return half_bits_to_float(static_cast<const std::uint16_t*>(values)[index]);
    return static_cast<const float*>(values)[index];
}

float round_value(float value, DType dtype) noexcept {
    return dtype == DType::fp16
        ? half_bits_to_float(float_to_half_bits(value))
        : value;
}

ProductForward product_forward(const ProblemView& problem, const GateView& gate) {
    require_product(problem);
    const auto cells = problem.batch;
    const auto coordinates = problem.topology.coordinates;
    const auto conditions = problem.topology.conditions;
    ProductForward result;
    result.y.resize(static_cast<std::size_t>(cells * coordinates));
    for (std::uint64_t b = 0; b < cells; ++b) {
        for (std::uint64_t x = 0; x < coordinates; ++x)
            result.y[static_cast<std::size_t>(b * coordinates + x)] =
                baseline_value(problem, b, x);
        for (std::uint64_t gi = 0; gi < problem.topology.group_count; ++gi) {
            const auto& group = problem.topology.groups[gi];
            if (!primal_active(gate, b, group.condition_index, conditions)) continue;
            const float q = condition_probability(gate, b, group.condition_index,
                                                  conditions);
            const float value = problem.theta[group.parameter_index] * q *
                                product_for(problem, b, group);
            for (std::uint64_t oi = 0; oi < group.output_count; ++oi) {
                const auto& incidence =
                    problem.topology.outputs[group.output_begin + oi];
                result.y[static_cast<std::size_t>(b * coordinates + incidence.coordinate)] +=
                    incidence.scale * value;
            }
        }
    }
    return result;
}

ProductBackward product_backward(const ProblemView& problem, const GateView& gate,
                                 const float* dy) {
    require_product(problem);
    if (problem.batch != 0 && problem.topology.coordinates != 0 && dy == nullptr)
        throw std::invalid_argument("product backward has no output gradient");
    const auto cells = problem.batch;
    const auto coordinates = problem.topology.coordinates;
    const auto conditions = problem.topology.conditions;
    ProductBackward result;
    result.dx.assign(static_cast<std::size_t>(cells * coordinates), 0.0f);
    result.dtheta.assign(static_cast<std::size_t>(problem.topology.parameters), 0.0f);
    result.dq.assign(static_cast<std::size_t>(cells * conditions), 0.0f);
    result.dscore.assign(static_cast<std::size_t>(cells * conditions), 0.0f);
    for (std::uint64_t b = 0; b < cells; ++b) {
        for (std::uint64_t gi = 0; gi < problem.topology.group_count; ++gi) {
            const auto& group = problem.topology.groups[gi];
            const auto condition = group.condition_index;
            if (!backward_active(gate, b, condition, conditions)) continue;
            const bool active = primal_active(gate, b, condition, conditions);
            if (!active && gate.profile != GateProfile::ste) continue;

            const float product = product_for(problem, b, group);
            float upstream = 0.0f;
            for (std::uint64_t oi = 0; oi < group.output_count; ++oi) {
                const auto& incidence =
                    problem.topology.outputs[group.output_begin + oi];
                upstream += dy[b * coordinates + incidence.coordinate] * incidence.scale;
            }
            const float theta = problem.theta[group.parameter_index];
            const float q = condition_probability(gate, b, condition, conditions);
            if (gate.profile == GateProfile::ste) {
                result.dq[static_cast<std::size_t>(b * conditions + condition)] +=
                    upstream * theta * product;
                const float p = gate.probabilities[b * conditions + condition];
                if (gate.revisions.temperature_bits == 0u)
                    throw std::invalid_argument("STE reference has zero temperature bits");
                float temperature = 0.0f;
                const auto temperature_bits = gate.revisions.temperature_bits;
                std::memcpy(&temperature, &temperature_bits, sizeof(temperature));
                if (!(temperature > 0.0f) || !std::isfinite(temperature))
                    throw std::invalid_argument("STE reference temperature must be finite and positive");
                result.dscore[static_cast<std::size_t>(b * conditions + condition)] +=
                    upstream * theta * product * p * (1.0f - p) / temperature;
            }
            if (!active) continue;
            result.dtheta[static_cast<std::size_t>(group.parameter_index)] +=
                upstream * q * product;
            const float factor = upstream * theta * q;
            for (std::uint64_t slot = 0; slot < group.argument_count; ++slot) {
                float other_product = 1.0f;
                for (std::uint64_t other = 0; other < group.argument_count; ++other) {
                    if (other == slot) continue;
                    const auto coordinate = problem.topology.arguments[
                        group.argument_begin + other];
                    other_product *= read_value(problem.x, problem.dtype,
                        b * coordinates + coordinate);
                }
                const auto coordinate =
                    problem.topology.arguments[group.argument_begin + slot];
                result.dx[static_cast<std::size_t>(b * coordinates + coordinate)] +=
                    factor * other_product;
            }
        }
    }
    return result;
}

LinearForward grouped_linear_forward(const ProblemView& problem,
                                     const GateView& gate) {
    require_linear(problem);
    const auto cells = problem.batch;
    const auto coordinates = problem.topology.coordinates;
    const auto conditions = problem.topology.conditions;
    const auto local_outputs = group_output_total(problem);
    LinearForward result;
    result.y.resize(static_cast<std::size_t>(cells * coordinates));
    result.group_f.assign(static_cast<std::size_t>(cells * local_outputs), 0.0f);
    for (std::uint64_t b = 0; b < cells; ++b) {
        for (std::uint64_t x = 0; x < coordinates; ++x)
            result.y[static_cast<std::size_t>(b * coordinates + x)] =
                baseline_value(problem, b, x);
        std::uint64_t f_offset = 0;
        for (std::uint64_t gi = 0; gi < problem.topology.group_count; ++gi) {
            const auto& group = problem.topology.groups[gi];
            if (!primal_active(gate, b, group.condition_index, conditions)) {
                f_offset += group.output_count;
                continue;
            }
            const float q = condition_probability(gate, b, group.condition_index,
                                                  conditions);
            const std::uint64_t weight_base =
                problem.weight_offsets[group.parameter_index];
            for (std::uint64_t out = 0; out < group.output_count; ++out) {
                float accumulator = 0.0f;
                for (std::uint64_t in = 0; in < group.argument_count; ++in) {
                    const auto coordinate = problem.topology.arguments[
                        group.argument_begin + in];
                    const auto weight_index = weight_base + out * group.argument_count + in;
                    accumulator += read_value(problem.x, problem.dtype,
                            b * coordinates + coordinate) *
                        read_value(problem.weights, problem.dtype, weight_index);
                }
                const float f = round_value(accumulator, problem.dtype);
                result.group_f[static_cast<std::size_t>(b * local_outputs + f_offset + out)] = f;
                const auto& incidence =
                    problem.topology.outputs[group.output_begin + out];
                result.y[static_cast<std::size_t>(b * coordinates + incidence.coordinate)] +=
                    incidence.scale * q * f;
            }
            f_offset += group.output_count;
        }
    }
    return result;
}

LinearBackward grouped_linear_backward(const ProblemView& problem,
                                      const GateView& gate, const float* dy,
                                      const LinearForward& forward) {
    require_linear(problem);
    if (problem.batch != 0 && problem.topology.coordinates != 0 && dy == nullptr)
        throw std::invalid_argument("linear backward has no output gradient");
    const auto cells = problem.batch;
    const auto coordinates = problem.topology.coordinates;
    const auto conditions = problem.topology.conditions;
    const auto local_outputs = group_output_total(problem);
    LinearBackward result;
    result.dx.assign(static_cast<std::size_t>(cells * coordinates), 0.0f);
    result.dweights.assign(static_cast<std::size_t>(problem.weight_count), 0.0f);
    result.dq.assign(static_cast<std::size_t>(cells * conditions), 0.0f);
    result.dscore.assign(static_cast<std::size_t>(cells * conditions), 0.0f);
    result.group_df.assign(static_cast<std::size_t>(cells * local_outputs), 0.0f);
    result.group_dx.assign(static_cast<std::size_t>(cells * problem.topology.argument_count), 0.0f);
    result.group_dw.assign(static_cast<std::size_t>(problem.group_weight_count), 0.0f);

    for (std::uint64_t gi = 0; gi < problem.topology.group_count; ++gi) {
        const auto& group = problem.topology.groups[gi];
        const auto condition = group.condition_index;
        const std::uint64_t weight_base = problem.weight_offsets[group.parameter_index];
        const std::uint64_t local_weight_base = problem.group_weight_offsets[gi];
        std::uint64_t f_offset = 0;
        for (std::uint64_t before = 0; before < gi; ++before)
            f_offset += problem.topology.groups[before].output_count;
        for (std::uint64_t b = 0; b < cells; ++b) {
            if (!backward_active(gate, b, condition, conditions)) continue;
            const bool active = primal_active(gate, b, condition, conditions);
            if (!active && gate.profile != GateProfile::ste) continue;
            const float q = active
                ? condition_probability(gate, b, condition, conditions) : 0.0f;
            const bool need_gate_derivative = gate.profile == GateProfile::ste;
            float dq = 0.0f;
            for (std::uint64_t out = 0; out < group.output_count; ++out) {
                if (need_gate_derivative) {
                    float f = 0.0f;
                    if (active) {
                        if (forward.group_f.size() !=
                            static_cast<std::size_t>(cells * local_outputs))
                            throw std::invalid_argument("linear backward forward activation extent mismatch");
                        f = forward.group_f[static_cast<std::size_t>(
                            b * local_outputs + f_offset + out)];
                    } else {
                        float accumulator = 0.0f;
                        for (std::uint64_t in = 0; in < group.argument_count; ++in) {
                            const auto coordinate = problem.topology.arguments[
                                group.argument_begin + in];
                            const auto weight_index = weight_base + out * group.argument_count + in;
                            accumulator += read_value(problem.x, problem.dtype,
                                    b * coordinates + coordinate) *
                                read_value(problem.weights, problem.dtype, weight_index);
                        }
                        f = round_value(accumulator, problem.dtype);
                    }
                    const auto& incidence =
                        problem.topology.outputs[group.output_begin + out];
                    const float grad = dy[b * coordinates + incidence.coordinate] *
                                       incidence.scale;
                    dq += grad * f;
                }
                if (!active) continue;
                const auto& incidence =
                    problem.topology.outputs[group.output_begin + out];
                const float grad = dy[b * coordinates + incidence.coordinate] *
                                   incidence.scale;
                result.group_df[static_cast<std::size_t>(
                    b * local_outputs + f_offset + out)] =
                    round_value(grad * q, problem.dtype);
            }
            if (need_gate_derivative) {
                result.dq[static_cast<std::size_t>(b * conditions + condition)] += dq;
                const float p = gate.probabilities[b * conditions + condition];
                if (gate.revisions.temperature_bits == 0u)
                    throw std::invalid_argument("STE reference has zero temperature bits");
                float temperature = 0.0f;
                const auto temperature_bits = gate.revisions.temperature_bits;
                std::memcpy(&temperature, &temperature_bits, sizeof(temperature));
                if (!(temperature > 0.0f) || !std::isfinite(temperature))
                    throw std::invalid_argument("STE reference temperature must be finite and positive");
                result.dscore[static_cast<std::size_t>(b * conditions + condition)] +=
                    dq * p * (1.0f - p) / temperature;
            }
            if (!active) continue;

            for (std::uint64_t in = 0; in < group.argument_count; ++in) {
                float accumulator = 0.0f;
                for (std::uint64_t out = 0; out < group.output_count; ++out) {
                    const auto weight_index = weight_base + out * group.argument_count + in;
                    accumulator += result.group_df[static_cast<std::size_t>(
                                       b * local_outputs + f_offset + out)] *
                                   read_value(problem.weights, problem.dtype, weight_index);
                }
                const float local_dx = round_value(accumulator, problem.dtype);
                const auto local_index = static_cast<std::size_t>(
                    b * problem.topology.argument_count + group.argument_begin + in);
                result.group_dx[local_index] = local_dx;
                const auto coordinate = problem.topology.arguments[
                    group.argument_begin + in];
                result.dx[static_cast<std::size_t>(b * coordinates + coordinate)] += local_dx;
            }

        }
        // dW is reduced once over the complete cell batch. Local FP16 dF
        // values above are consumed as stored values, accumulated in FP32,
        // and rounded once at the local dW boundary before shared-weight
        // canonical reduction.
        for (std::uint64_t out = 0; out < group.output_count; ++out) {
            for (std::uint64_t in = 0; in < group.argument_count; ++in) {
                float accumulator = 0.0f;
                const auto coordinate = problem.topology.arguments[
                    group.argument_begin + in];
                for (std::uint64_t row = 0; row < cells; ++row) {
                    if (!primal_active(gate, row, condition, conditions)) continue;
                    const float local_df = result.group_df[static_cast<std::size_t>(
                        row * local_outputs + f_offset + out)];
                    accumulator += local_df * read_value(problem.x, problem.dtype,
                        row * coordinates + coordinate);
                }
                const float local_dw = round_value(accumulator, problem.dtype);
                const auto local_index = static_cast<std::size_t>(
                    local_weight_base + out * group.argument_count + in);
                if (local_index >= result.group_dw.size())
                    throw std::invalid_argument("group weight offset exceeds reference storage");
                result.group_dw[local_index] = local_dw;
                const auto canonical = weight_base + out * group.argument_count + in;
                if (canonical >= result.dweights.size())
                    throw std::invalid_argument("parameter weight offset exceeds reference storage");
                result.dweights[static_cast<std::size_t>(canonical)] += local_dw;
            }
        }
    }
    (void) forward;
    return result;
}

float grouped_linear_allowed_error(DType dtype, float expected,
                                   std::size_t dot_length,
                                   double sum_abs_products) noexcept {
    const double unit_roundoff = std::numeric_limits<float>::epsilon() / 2.0;
    const double ku = 2.0 * static_cast<double>(dot_length) * unit_roundoff;
    const float accumulation = static_cast<float>(
        (2.0 * ku / (1.0 - ku)) * sum_abs_products);
    return accumulation + (dtype == DType::fp16 ? fp16_ulp(expected) : 0.0f);
}

namespace {

float fp32_reduction_bound(std::size_t terms, double sum_abs) noexcept {
    const double unit_roundoff = std::numeric_limits<float>::epsilon() / 2.0;
    const double ku = 2.0 * static_cast<double>(terms) * unit_roundoff;
    if (ku >= 1.0) return std::numeric_limits<float>::infinity();
    return static_cast<float>((2.0 * ku / (1.0 - ku)) * sum_abs);
}

float local_f_error(const ProblemView& problem, std::uint64_t batch,
                    const cellerator::ops::mechanism::GatedProductGroup& group,
                    std::uint64_t output, std::uint64_t weight_base,
                    float expected) {
    double sum_abs = 0.0;
    const auto coordinates = problem.topology.coordinates;
    for (std::uint64_t in = 0; in < group.argument_count; ++in) {
        const auto coordinate = problem.topology.arguments[group.argument_begin + in];
        const auto weight = weight_base + output * group.argument_count + in;
        sum_abs += std::abs(static_cast<double>(
            read_value(problem.x, problem.dtype, batch * coordinates + coordinate)) *
            read_value(problem.weights, problem.dtype, weight));
    }
    return grouped_linear_allowed_error(problem.dtype, expected,
                                        group.argument_count, sum_abs);
}

} // namespace

LinearValidationBounds grouped_linear_validation_bounds(
    const ProblemView& problem, const GateView& gate, const float* dy,
    const LinearForward& forward, const LinearBackward& backward) {
    require_linear(problem);
    const auto cells = problem.batch;
    const auto coordinates = problem.topology.coordinates;
    const auto conditions = problem.topology.conditions;
    const auto local_outputs = group_output_total(problem);
    if (forward.group_f.size() != static_cast<std::size_t>(cells * local_outputs) ||
        backward.group_df.size() != static_cast<std::size_t>(cells * local_outputs) ||
        backward.group_dx.size() != static_cast<std::size_t>(
            cells * problem.topology.argument_count) ||
        backward.group_dw.size() != problem.group_weight_count)
        throw std::invalid_argument("linear validation reference extents disagree");
    if (gate.profile == GateProfile::ste && gate.probabilities == nullptr)
        throw std::invalid_argument("STE validation requires probabilities");
    LinearValidationBounds bounds;
    bounds.y.assign(static_cast<std::size_t>(cells * coordinates), 0.0f);
    bounds.dx.assign(static_cast<std::size_t>(cells * coordinates), 0.0f);
    bounds.dweights.assign(static_cast<std::size_t>(problem.weight_count), 0.0f);
    bounds.dq.assign(static_cast<std::size_t>(cells * conditions), 0.0f);
    bounds.dscore.assign(static_cast<std::size_t>(cells * conditions), 0.0f);
    bounds.group_dx.assign(static_cast<std::size_t>(cells * problem.topology.argument_count), 0.0f);
    bounds.group_dw.assign(static_cast<std::size_t>(problem.group_weight_count), 0.0f);
    std::vector<double> y_abs(bounds.y.size(), 0.0);
    std::vector<std::size_t> y_terms(bounds.y.size(), 0u);
    std::vector<double> y_local_error(bounds.y.size(), 0.0);
    std::vector<double> dx_abs(bounds.dx.size(), 0.0);
    std::vector<std::size_t> dx_terms(bounds.dx.size(), 0u);
    std::vector<double> dx_local_error(bounds.dx.size(), 0.0);
    std::vector<double> dw_abs(bounds.dweights.size(), 0.0);
    std::vector<std::size_t> dw_terms(bounds.dweights.size(), 0u);
    std::vector<double> dw_local_error(bounds.dweights.size(), 0.0);
    std::vector<double> dq_abs(bounds.dq.size(), 0.0);
    std::vector<std::size_t> dq_terms(bounds.dq.size(), 0u);
    std::vector<double> dq_local_error(bounds.dq.size(), 0.0);

    for (std::uint64_t b = 0; b < cells; ++b) {
        if (problem.include_baseline) {
            for (std::uint64_t coordinate = 0; coordinate < coordinates; ++coordinate) {
                const auto at = static_cast<std::size_t>(b * coordinates + coordinate);
                const float value = problem.beta * read_value(problem.x, problem.dtype,
                    b * coordinates + coordinate);
                y_abs[at] += std::abs(static_cast<double>(value));
                y_local_error[at] += fp32_reduction_bound(1, std::abs(value));
                ++y_terms[at];
            }
        }
        std::uint64_t f_offset = 0;
        for (std::uint64_t gi = 0; gi < problem.topology.group_count; ++gi) {
            const auto& group = problem.topology.groups[gi];
            if (!primal_active(gate, b, group.condition_index, conditions)) {
                f_offset += group.output_count;
                continue;
            }
            const float q = condition_probability(gate, b, group.condition_index,
                                                  conditions);
            const auto weight_base = problem.weight_offsets[group.parameter_index];
            for (std::uint64_t out = 0; out < group.output_count; ++out) {
                const float f = forward.group_f[static_cast<std::size_t>(
                    b * local_outputs + f_offset + out)];
                const float f_error = local_f_error(problem, b, group, out,
                                                    weight_base, f);
                const auto& incidence =
                    problem.topology.outputs[group.output_begin + out];
                const float term = incidence.scale * q * f;
                const auto at = static_cast<std::size_t>(
                    b * coordinates + incidence.coordinate);
                y_abs[at] += std::abs(static_cast<double>(term));
                y_local_error[at] += std::abs(static_cast<double>(incidence.scale * q)) *
                    f_error + fp32_reduction_bound(2, std::abs(static_cast<double>(term)));
                ++y_terms[at];
            }
            f_offset += group.output_count;
        }
    }

    for (std::uint64_t gi = 0; gi < problem.topology.group_count; ++gi) {
        const auto& group = problem.topology.groups[gi];
        const auto condition = group.condition_index;
        const auto weight_base = problem.weight_offsets[group.parameter_index];
        const auto local_weight_base = problem.group_weight_offsets[gi];
        std::uint64_t f_offset = 0;
        for (std::uint64_t before = 0; before < gi; ++before)
            f_offset += problem.topology.groups[before].output_count;

        if (gate.profile == GateProfile::ste) {
            for (std::uint64_t b = 0; b < cells; ++b) {
                const bool active = primal_active(gate, b, condition, conditions);
                if (!backward_active(gate, b, condition, conditions)) continue;
                float dq_group_abs = 0.0f;
                float dq_group_local_error = 0.0f;
                std::size_t dq_group_terms = 0;
                for (std::uint64_t out = 0; out < group.output_count; ++out) {
                    float f = 0.0f;
                    if (active) {
                        f = forward.group_f[static_cast<std::size_t>(
                            b * local_outputs + f_offset + out)];
                    } else {
                        float accumulator = 0.0f;
                        for (std::uint64_t in = 0; in < group.argument_count; ++in) {
                            const auto coordinate = problem.topology.arguments[
                                group.argument_begin + in];
                            accumulator += read_value(problem.x, problem.dtype,
                                b * coordinates + coordinate) *
                                read_value(problem.weights, problem.dtype,
                                    weight_base + out * group.argument_count + in);
                        }
                        f = round_value(accumulator, problem.dtype);
                    }
                    const float f_error = local_f_error(problem, b, group, out,
                                                        weight_base, f);
                    const auto& incidence =
                        problem.topology.outputs[group.output_begin + out];
                    const float grad = dy[b * coordinates + incidence.coordinate] *
                                       incidence.scale;
                    const float term = grad * f;
                    dq_group_abs += std::abs(term);
                    dq_group_local_error += std::abs(static_cast<double>(grad)) * f_error +
                        fp32_reduction_bound(2, std::abs(static_cast<double>(term)));
                    ++dq_group_terms;
                }
                const auto at = static_cast<std::size_t>(b * conditions + condition);
                dq_abs[at] += dq_group_abs;
                dq_local_error[at] += dq_group_local_error;
                dq_terms[at] += dq_group_terms;
            }
        }

        for (std::uint64_t b = 0; b < cells; ++b) {
            if (!primal_active(gate, b, condition, conditions)) continue;
            for (std::uint64_t in = 0; in < group.argument_count; ++in) {
                double sum_abs = 0.0;
                for (std::uint64_t out = 0; out < group.output_count; ++out) {
                    const float local_df = backward.group_df[static_cast<std::size_t>(
                        b * local_outputs + f_offset + out)];
                    sum_abs += std::abs(static_cast<double>(local_df) *
                        read_value(problem.weights, problem.dtype,
                            weight_base + out * group.argument_count + in));
                }
                const auto local_index = static_cast<std::size_t>(
                    b * problem.topology.argument_count + group.argument_begin + in);
                const float local_expected = backward.group_dx[local_index];
                bounds.group_dx[local_index] = grouped_linear_allowed_error(
                    problem.dtype, local_expected, group.output_count, sum_abs);
                const auto coordinate = problem.topology.arguments[
                    group.argument_begin + in];
                const auto canonical = static_cast<std::size_t>(
                    b * coordinates + coordinate);
                dx_abs[canonical] += std::abs(static_cast<double>(local_expected));
                dx_local_error[canonical] += bounds.group_dx[local_index];
                ++dx_terms[canonical];
            }
        }

        for (std::uint64_t out = 0; out < group.output_count; ++out) {
            for (std::uint64_t in = 0; in < group.argument_count; ++in) {
                double sum_abs = 0.0;
                std::size_t rows = static_cast<std::size_t>(cells);
                const auto coordinate = problem.topology.arguments[
                    group.argument_begin + in];
                for (std::uint64_t b = 0; b < cells; ++b) {
                    if (!primal_active(gate, b, condition, conditions)) continue;
                    const float local_df = backward.group_df[static_cast<std::size_t>(
                        b * local_outputs + f_offset + out)];
                    sum_abs += std::abs(static_cast<double>(local_df) *
                        read_value(problem.x, problem.dtype,
                                   b * coordinates + coordinate));
                }
                const auto local_index = static_cast<std::size_t>(
                    local_weight_base + out * group.argument_count + in);
                if (local_index >= backward.group_dw.size())
                    throw std::invalid_argument("local dW reference extent mismatch");
                const float local_expected = backward.group_dw[local_index];
                bounds.group_dw[local_index] = grouped_linear_allowed_error(
                    problem.dtype, local_expected, rows, sum_abs);
                const auto canonical = static_cast<std::size_t>(
                    weight_base + out * group.argument_count + in);
                if (canonical >= backward.dweights.size())
                    throw std::invalid_argument("canonical dW reference extent mismatch");
                dw_abs[canonical] += std::abs(static_cast<double>(local_expected));
                dw_local_error[canonical] += bounds.group_dw[local_index];
                ++dw_terms[canonical];
            }
        }
    }

    for (std::size_t i = 0; i < bounds.y.size(); ++i)
        bounds.y[i] = static_cast<float>(y_local_error[i] +
            fp32_reduction_bound(y_terms[i], y_abs[i]));
    for (std::size_t i = 0; i < bounds.dx.size(); ++i)
        bounds.dx[i] = static_cast<float>(dx_local_error[i] +
            fp32_reduction_bound(dx_terms[i], dx_abs[i]));
    for (std::size_t i = 0; i < bounds.dweights.size(); ++i)
        bounds.dweights[i] = static_cast<float>(dw_local_error[i] +
            fp32_reduction_bound(dw_terms[i], dw_abs[i]));
    if (gate.profile == GateProfile::ste) {
        for (std::size_t i = 0; i < bounds.dq.size(); ++i)
            bounds.dq[i] = static_cast<float>(dq_local_error[i] +
                fp32_reduction_bound(dq_terms[i], dq_abs[i]));
        float temperature = 0.0f;
        std::memcpy(&temperature, &gate.revisions.temperature_bits, sizeof(temperature));
        if (!(temperature > 0.0f) || !std::isfinite(temperature))
            throw std::invalid_argument("STE validation needs a finite positive temperature");
        for (std::uint64_t b = 0; b < cells; ++b)
            for (std::uint64_t c = 0; c < conditions; ++c) {
                if (!derivative_bit(gate, b, c, conditions)) continue;
                const auto at = static_cast<std::size_t>(b * conditions + c);
                const float p = gate.probabilities[at];
                const float chain = p * (1.0f - p) / temperature;
                bounds.dscore[at] = std::abs(chain) * bounds.dq[at] +
                    fp32_reduction_bound(4,
                        std::abs(static_cast<double>(backward.dscore[at])));
            }
    }
    return bounds;
}

bool within_absolute_bound(float actual, float expected,
                           float absolute_bound) noexcept {
    return std::isfinite(actual) && std::isfinite(expected) &&
           std::isfinite(absolute_bound) && absolute_bound >= 0.0f &&
           std::abs(actual - expected) <= absolute_bound;
}

TeacherExampleReference teacher_example_reference(float logit,
                                                   float label) {
    if (!(label >= 0.0f && label <= 1.0f) || !std::isfinite(label))
        throw std::invalid_argument("teacher Bernoulli label must be finite in [0,1]");
    const float probability = logit >= 0.0f
        ? 1.0f / (1.0f + std::exp(-logit))
        : std::exp(logit) / (1.0f + std::exp(logit));
    const float bce = std::max(logit, 0.0f) - logit * label +
                      std::log1p(std::exp(-std::abs(logit)));
    return {probability, bce, probability - label};
}

TeacherReference teacher_reference(const std::array<float, 8>& logits,
                                   const std::array<float, 8>& labels) {
    std::array<std::uint64_t, 8> counts{};
    std::array<double, 8> positives{};
    counts.fill(1u);
    for (std::size_t i = 0; i < labels.size(); ++i) positives[i] = labels[i];
    return teacher_row_mean_reference(logits, counts, positives);
}

TeacherReference teacher_batch_reference(const std::array<float, 8>& logits,
                                         const std::uint8_t* rows,
                                         const float* labels,
                                         std::size_t count) {
    if (count != 0 && (rows == nullptr || labels == nullptr))
        throw std::invalid_argument("teacher batch reference has null examples");
    TeacherReference result;
    for (std::size_t row = 0; row < logits.size(); ++row) {
        const auto value = teacher_example_reference(logits[row], 0.0f);
        result.probability[row] = value.probability;
        result.decision[row] = value.probability >= 0.5f ? 1u : 0u;
        if (result.decision[row]) result.table |= static_cast<std::uint8_t>(1u << row);
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (rows[i] >= logits.size())
            throw std::invalid_argument("teacher sample row must be in [0,7]");
        const auto value = teacher_example_reference(logits[rows[i]], labels[i]);
        result.mean_bce += value.bce;
        result.mean_logit_gradient[rows[i]] += value.dlogit;
    }
    if (count != 0) {
        const float inv = 1.0f / static_cast<float>(count);
        result.mean_bce *= inv;
        for (float& gradient : result.mean_logit_gradient) gradient *= inv;
    }
    return result;
}

TeacherReference teacher_row_mean_reference(
    const std::array<float, 8>& logits,
    const std::array<std::uint64_t, 8>& row_counts,
    const std::array<double, 8>& positive_counts) {
    TeacherReference result;
    std::size_t populated = 0;
    for (std::size_t row = 0; row < logits.size(); ++row) {
        const auto unlabeled = teacher_example_reference(logits[row], 0.0f);
        result.probability[row] = unlabeled.probability;
        result.decision[row] = unlabeled.probability >= 0.5f ? 1u : 0u;
        if (result.decision[row]) result.table |= static_cast<std::uint8_t>(1u << row);
        if (row_counts[row] == 0u) {
            if (positive_counts[row] != 0.0)
                throw std::invalid_argument("empty teacher row has nonzero positive count");
            continue;
        }
        if (!std::isfinite(positive_counts[row]) || positive_counts[row] < 0.0 ||
            positive_counts[row] > static_cast<double>(row_counts[row]))
            throw std::invalid_argument("teacher positive count is outside row count");
        const float target = static_cast<float>(
            positive_counts[row] / static_cast<double>(row_counts[row]));
        const auto value = teacher_example_reference(logits[row], target);
        result.per_row_bce[row] = value.bce;
        result.mean_logit_gradient[row] = value.dlogit;
        result.row_has_examples[row] = 1u;
        result.mean_bce += value.bce;
        ++populated;
    }
    if (populated != 0)
        result.mean_bce /= static_cast<float>(populated);
    return result;
}

std::uint8_t teacher_lookup(std::uint8_t table, bool a, bool b,
                            bool c) noexcept {
    const std::uint8_t row = static_cast<std::uint8_t>(
        (a ? 4u : 0u) | (b ? 2u : 0u) | (c ? 1u : 0u));
    return static_cast<std::uint8_t>((table >> row) & 1u);
}

} // namespace cgs::reference
