#include <Cellerator/compute/operation/relation_algebra.hh>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {

using exchange_operation_function_v1 = bool (*)(
    void *context, void *stream, std::uint64_t input_generation,
    std::uint64_t output_generation) noexcept;

enum class exchange_program_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    operation_failure = 2u
};

struct prepared_exchange_step_v1 {
    compute::operation::relation_algebra_kind_v1 kind =
        compute::operation::relation_algebra_kind_v1::contract_on_support;
    exchange_operation_function_v1 execute = nullptr;
    void *context = nullptr;
    std::uint64_t input_generation = 0u;
    std::uint64_t output_generation = 0u;
};

struct prepared_exchange_program_v1 {
    std::uint32_t schema_version = 1u;
    std::uint32_t step_count = 0u;
    const prepared_exchange_step_v1 *steps = nullptr;
    void *stream = nullptr;
};

exchange_program_status_v1 run_prepared_exchange_program_v1(
    const prepared_exchange_program_v1 &program) noexcept {
    using kind = compute::operation::relation_algebra_kind_v1;
    constexpr kind required[] = {kind::contract_on_support,
        kind::edge_map_or_gate, kind::segment_normalize,
        kind::relation_apply};
    if (program.schema_version != 1u || program.step_count != 4u
        || program.steps == nullptr || program.stream == nullptr)
        return exchange_program_status_v1::invalid_argument;
    for (std::uint32_t index = 0u; index < program.step_count; ++index) {
        const prepared_exchange_step_v1 &step = program.steps[index];
        if (step.kind != required[index] || step.execute == nullptr
            || step.context == nullptr || step.input_generation == 0u
            || step.output_generation == 0u
            || (index != 0u && program.steps[index - 1u].output_generation
                != step.input_generation))
            return exchange_program_status_v1::invalid_argument;
    }
    for (std::uint32_t index = 0u; index < program.step_count; ++index) {
        const prepared_exchange_step_v1 &step = program.steps[index];
        if (!step.execute(step.context, program.stream, step.input_generation,
                step.output_generation))
            return exchange_program_status_v1::operation_failure;
    }
    return exchange_program_status_v1::success;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
