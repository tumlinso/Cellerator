#include "../../../src/compute/architecture/providers/nvidia/sm70/exchange_program.cc"

#include <cassert>
#include <cstdint>

namespace operation = cellerator::compute::operation;
namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;

namespace {
struct trace { std::uint32_t calls = 0u; std::uint64_t last = 0u; bool fail = false; };
bool execute(void *opaque, void *stream, std::uint64_t input,
    std::uint64_t output) noexcept {
    auto *state = static_cast<trace *>(opaque);
    if (stream == nullptr || input != state->last) return false;
    ++state->calls;
    state->last = output;
    return !state->fail;
}
}

int main() {
    using kind = operation::relation_algebra_kind_v1;
    trace state{};
    state.last = 7u;
    sm70::prepared_exchange_step_v1 steps[4]{};
    const kind kinds[] = {kind::contract_on_support, kind::edge_map_or_gate,
        kind::segment_normalize, kind::relation_apply};
    for (std::uint32_t index = 0u; index < 4u; ++index) {
        steps[index].kind = kinds[index];
        steps[index].execute = &execute;
        steps[index].context = &state;
        steps[index].input_generation = 7u + index;
        steps[index].output_generation = 8u + index;
    }
    int stream = 1;
    sm70::prepared_exchange_program_v1 program{};
    program.step_count = 4u;
    program.steps = steps;
    program.stream = &stream;
    assert(sm70::run_prepared_exchange_program_v1(program)
        == sm70::exchange_program_status_v1::success);
    assert(state.calls == 4u && state.last == 11u);
    steps[2].input_generation = 99u;
    assert(sm70::run_prepared_exchange_program_v1(program)
        == sm70::exchange_program_status_v1::invalid_argument);
    steps[2].input_generation = 9u;
    state = {0u, 7u, true};
    assert(sm70::run_prepared_exchange_program_v1(program)
        == sm70::exchange_program_status_v1::operation_failure);
    return 0;
}
