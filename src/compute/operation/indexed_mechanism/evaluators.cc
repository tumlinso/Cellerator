#include <Cellerator/compute/operation/indexed_mechanism/evaluators.hh>

#include <cmath>

namespace cellerator::compute::operation::indexed {
namespace {
bool valid_opcode(evaluator_opcode opcode) noexcept {
    return opcode >= evaluator_opcode::sum && opcode <= evaluator_opcode::first_minus_product_tail;
}
bool valid_block(const registered_block& block) noexcept {
    return v2::valid_stable_id(block.evaluator) && valid_opcode(block.opcode)
        && block.dependencies_known && block.writes_only_declared_outputs && block.deterministic;
}
template <class T> float decode(T value) noexcept { return static_cast<float>(value); }
template <> float decode<std::uint16_t>(std::uint16_t value) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16u;
    std::uint32_t exponent = (value >> 10u) & 0x1fu;
    std::uint32_t fraction = value & 0x03ffu;
    if (exponent == 0) {
        if (fraction == 0) { union { std::uint32_t u; float f; } bits{sign}; return bits.f; }
        exponent = 113;
        while ((fraction & 0x0400u) == 0) { fraction <<= 1u; --exponent; }
        fraction &= 0x03ffu;
    } else if (exponent == 31) {
        union { std::uint32_t u; float f; } bits{sign | 0x7f800000u | (fraction << 13u)}; return bits.f;
    } else exponent += 112;
    union { std::uint32_t u; float f; } bits{sign | (exponent << 23u) | (fraction << 13u)}; return bits.f;
}
std::uint16_t encode_half_rne(float value) noexcept {
    union { float f; std::uint32_t u; } bits{value};
    const std::uint32_t sign = (bits.u >> 16u) & 0x8000u;
    const std::uint32_t magnitude = bits.u & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) return static_cast<std::uint16_t>(sign | (magnitude == 0x7f800000u ? 0x7c00u : 0x7e00u));
    std::uint32_t exponent = magnitude >> 23u;
    std::uint32_t mantissa = magnitude & 0x7fffffu;
    if (exponent > 142u) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (exponent < 103u) return static_cast<std::uint16_t>(sign);
    if (exponent < 113u) {
        const std::uint32_t shift = 126u - exponent;
        mantissa = (mantissa | 0x800000u);
        const std::uint32_t result = (mantissa + (1u << (shift - 1u)) - 1u
            + ((mantissa >> shift) & 1u)) >> shift;
        return static_cast<std::uint16_t>(sign | result);
    }
    std::uint32_t half_exponent = exponent - 112u;
    mantissa = (mantissa + 0x0fffu + ((mantissa >> 13u) & 1u)) >> 13u;
    if (mantissa == 0x0400u) { mantissa = 0; ++half_exponent; }
    if (half_exponent >= 31u) return static_cast<std::uint16_t>(sign | 0x7c00u);
    return static_cast<std::uint16_t>(sign | (half_exponent << 10u) | mantissa);
}
template <class T> evaluation_status evaluate(const registered_block& block, bool predicate,
                                              std::span<const T> arguments, std::span<T> destination) noexcept {
    if (!valid_block(block)) return evaluation_status::invalid_block;
    if (arguments.size() < block.minimum_arguments) return evaluation_status::invalid_arity;
    if (destination.size() != 1 || (!arguments.empty() && !arguments.data()) || !destination.data())
        return evaluation_status::invalid_binding;
    if (!predicate) return evaluation_status::success;
    float result = 0.0f;
    switch (block.opcode) {
    case evaluator_opcode::sum:
        for (const auto value : arguments) result += decode(value);
        break;
    case evaluator_opcode::product:
        result = 1.0f;
        for (const auto value : arguments) result *= decode(value);
        break;
    case evaluator_opcode::first_minus_product_tail:
        if (arguments.size() < 3) return evaluation_status::invalid_arity;
        result = 1.0f;
        for (std::size_t i = 1; i < arguments.size(); ++i) result *= decode(arguments[i]);
        result = decode(arguments[0]) - result;
        break;
    }
    if constexpr (std::is_same_v<T, std::uint16_t>) destination[0] = encode_half_rne(result);
    else destination[0] = result;
    return evaluation_status::success;
}
}

evaluation_status block_registry::register_block(registered_block block) noexcept {
    if (!valid_block(block)) return evaluation_status::invalid_block;
    for (const auto& current : blocks_)
        if (v2::same_stable_id(current.evaluator, block.evaluator)) return evaluation_status::duplicate_block;
    try { blocks_.push_back(block); return evaluation_status::success; }
    catch (...) { return evaluation_status::invalid_binding; }
}
const registered_block* block_registry::find(identity evaluator) const noexcept {
    for (const auto& block : blocks_) if (v2::same_stable_id(block.evaluator, evaluator)) return &block;
    return nullptr;
}
bool valid_prepared_stage(const prepared_evaluator_stage& stage) noexcept {
    return valid_block(stage.block) && stage.argument_count >= stage.block.minimum_arguments
        && (stage.block.opcode != evaluator_opcode::first_minus_product_tail || stage.argument_count >= 3)
        && stage.required_value_generation != 0;
}
execution::program::program_status launch_evaluator_stage(
        const void* prepared_state, const execution::program::launch_binding_v2& binding,
        void* caller_stream) noexcept {
    if (admit_evaluator_stage(prepared_state, binding, caller_stream) !=
        execution::program::program_status::success)
        return execution::program::program_status::invalid_argument;
    const auto* stage = static_cast<const prepared_evaluator_stage*>(prepared_state);
    const auto* values = static_cast<const evaluator_stage_values*>(binding.values);
    return evaluate_cuda_f32(stage->block, values->predicate,
        static_cast<const float*>(binding.input), stage->argument_count,
        static_cast<float*>(binding.output), caller_stream) == evaluation_status::success
        ? execution::program::program_status::success
        : execution::program::program_status::launch_failed;
}
execution::program::program_status admit_evaluator_stage(
        const void* prepared_state, const execution::program::launch_binding_v2& binding,
        void*) noexcept {
    const auto* stage = static_cast<const prepared_evaluator_stage*>(prepared_state);
    const auto* values = static_cast<const evaluator_stage_values*>(binding.values);
    if (!stage || !values || values->binding_magic != evaluator_stage_values::magic
        || values->value_generation != stage->required_value_generation
        || !binding.input || !binding.output || !valid_prepared_stage(*stage))
        return execution::program::program_status::invalid_argument;
    return execution::program::program_status::success;
}
execution::program::prepared_stage_v2 make_prepared_stage(
        std::uint64_t stable_stage_id, std::uint64_t candidate_id,
        const prepared_evaluator_stage& state, std::uint32_t binding_index,
        std::uint64_t first_dependency, std::uint32_t dependency_count) noexcept {
    return {stable_stage_id, candidate_id, &state, launch_evaluator_stage, first_dependency,
            dependency_count, binding_index, 0, admit_evaluator_stage};
}
evaluation_status evaluate_f32(const registered_block& block, bool predicate,
                               std::span<const float> arguments, std::span<float> destination) noexcept {
    return evaluate(block, predicate, arguments, destination);
}
evaluation_status evaluate_f16(const registered_block& block, bool predicate,
                               std::span<const std::uint16_t> arguments, std::span<std::uint16_t> destination) noexcept {
    return evaluate(block, predicate, arguments, destination);
}
} // namespace cellerator::compute::operation::indexed
