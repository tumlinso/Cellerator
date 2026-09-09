#include <Cellerator/execution/program/program_v2.h>
#include <Cellerator/execution/joint_compiler/external_binding_v1.hh>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace ce = cellerator::execution;
namespace pg = ce::program;
namespace jc = ce::joint_compiler;

void check(bool condition) {
    if (!condition) std::abort();
}

struct add_contract {
    std::uint64_t count;
    ce::order_id order;
};
struct add_bindings {
    jc::external_binding_v1 input;
    jc::external_binding_v1 output;
};
unsigned enqueues = 0;

// A numerical owner's pure preflight composes the existing extent validator.
pg::program_status preflight_add(const void* state,
        const pg::launch_binding_v2& binding, void*) noexcept {
    if (!state || !binding.validation_state)
        return pg::program_status::invalid_argument;
    const auto& contract = *static_cast<const add_contract*>(state);
    const auto& bound = *static_cast<const add_bindings*>(binding.validation_state);
    if (!jc::validate_external_binding_v1(bound.input) ||
        !jc::validate_external_binding_v1(bound.output) ||
        bound.input.extent_count != 1 || bound.output.extent_count != 1 ||
        contract.count > std::numeric_limits<std::uint64_t>::max() / sizeof(float))
        return pg::program_status::invalid_argument;
    const auto& input = bound.input.extents[0];
    const auto& output = bound.output.extents[0];
    const auto bytes = contract.count * sizeof(float);
    if (input.bytes < bytes || output.bytes < bytes ||
        input.address != binding.input || output.address != binding.output ||
        !ce::same_identity(input.order, contract.order) ||
        !ce::same_identity(output.order, contract.order) ||
        input.location.residency != ce::residency_kind::host ||
        output.location.residency != ce::residency_kind::host)
        return pg::program_status::invalid_argument;
    const auto begin = reinterpret_cast<std::uintptr_t>(input.address);
    const auto destination = reinterpret_cast<std::uintptr_t>(output.address);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - destination ||
        (begin < destination + bytes && destination < begin + bytes))
        return pg::program_status::invalid_argument;
    return pg::program_status::success;
}

pg::program_status add(const void* state,
        const pg::launch_binding_v2& binding, void*) noexcept {
    ++enqueues;
    const auto& contract = *static_cast<const add_contract*>(state);
    const auto* input = static_cast<const float*>(binding.input);
    auto* output = static_cast<float*>(binding.output);
    for (std::uint64_t i = 0; i < contract.count; ++i) output[i] = input[i] + 1;
    return pg::program_status::success;
}

jc::external_extent_v1 extent(void* address) {
    jc::external_extent_v1 value{};
    value.address = address;
    value.location = {ce::residency_kind::host, {}, -1, 1};
    value.bytes = 4 * sizeof(float);
    value.alignment = alignof(float);
    value.order = {1, 1};
    value.generation = {1};
    value.readiness = {1, 1};
    value.lease = {2, 1};
    return value;
}
jc::external_binding_v1 external(const jc::external_extent_v1* value) {
    jc::external_binding_v1 result{};
    result.binding_identity = {1, 1};
    result.atom_identity = {2, 1};
    result.plane_identity = {3, 1};
    result.extents = value;
    result.extent_count = 1;
    result.total_bytes = value->bytes;
    return result;
}

int main() {
    std::array<float, 4> input{1, 2, 3, 4}, middle{}, output{};
    auto input_extent = extent(input.data());
    auto middle_extent = extent(middle.data());
    auto output_extent = extent(output.data());
    add_bindings data[]{{external(&input_extent), external(&middle_extent)},
                        {external(&middle_extent), external(&output_extent)}};
    add_contract contract{4, {1, 1}};
    pg::prepared_stage_v2 stages[]{
        {1, 1, &contract, add, 0, 0, 0, 0, nullptr, preflight_add},
        {2, 1, &contract, add, 0, 1, 1, 0, nullptr, preflight_add}};
    std::uint64_t dependency = 0;
    pg::prepared_program_v2 program{2, 0, stages, 2, &dependency, 1};
    pg::launch_binding_v2 bindings[]{
        {input.data(), middle.data(), nullptr, nullptr, 0, nullptr, &data[0]},
        {middle.data(), output.data(), nullptr, nullptr, 0, nullptr, &data[1]}};
    auto execute = [&] { return pg::execute_prepared_program_v2(program, bindings, 2, nullptr); };
    auto rejected = [&](pg::program_status expected) {
        check(execute() == expected);
        check(enqueues == 0);
        check(middle == std::array<float, 4>{} && output == std::array<float, 4>{});
    };
    // Only the final binding is invalid; the valid first stage cannot execute.
    output_extent.bytes = sizeof(float);
    data[1].output.total_bytes = sizeof(float);
    rejected(pg::program_status::invalid_dynamic_binding);
    output_extent.bytes = sizeof(output);
    data[1].output.total_bytes = sizeof(output);
    output_extent.order = {9, 1};
    rejected(pg::program_status::invalid_dynamic_binding);
    output_extent.order = {1, 1};
    output_extent.address = middle.data() + 1;
    bindings[1].output = middle.data() + 1;
    rejected(pg::program_status::invalid_dynamic_binding);
    output_extent.address = output.data(); bindings[1].output = output.data();
    bindings[1].validation_state = nullptr;
    rejected(pg::program_status::invalid_dynamic_binding);
    bindings[1].validation_state = &data[1];
    stages[1].required_workspace_bytes = 16;
    rejected(pg::program_status::insufficient_bindings);
    stages[1].required_workspace_bytes = 0;
    stages[1].binding_index = 2;
    rejected(pg::program_status::insufficient_bindings);
    stages[1].binding_index = 1;
    check(pg::preflight_prepared_program_v2(program, bindings, 2, nullptr) == pg::program_status::success);
    check(enqueues == 0);
    check(execute() == pg::program_status::success && enqueues == 2);
    for (unsigned i = 0; i < 4; ++i) check(output[i] == input[i] + 2);
    // Revalidation on every launch; a previously accepted pointer is no cache key.
    middle.fill(0); output.fill(0); enqueues = 0;
    output_extent.bytes = 1;
    rejected(pg::program_status::invalid_dynamic_binding);
    std::cout << "all-stage preflight, external bounds, range aliases and repeated launch: passed\n";
}
