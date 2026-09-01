#include <Cellerator/execution/atom_fragment/compiler_registry_v1.hh>

#include <cassert>

namespace fragment = cellerator::execution::atom_fragment;
namespace joint = cellerator::execution::joint_compiler;
namespace program = cellerator::execution::program;

fragment::fragment_compile_status_code_v1 compile(const void *context,
    const joint::atom_fragment_request_v1 &, std::uint64_t candidate_id,
    program::prepared_program_v2 *output) noexcept {
    if (context == nullptr || candidate_id == 0u || output == nullptr)
        return fragment::fragment_compile_status_code_v1::invalid_request;
    output->flags = *static_cast<const std::uint32_t *>(context);
    return fragment::fragment_compile_status_code_v1::success;
}

int main() {
    const std::uint32_t contexts[] = {10u, 20u, 30u};
    fragment::fragment_compiler_entry_v1 entries[] = {
        {{1u, 1u}, 7u, {10u, 1u}, &contexts[0], compile},
        {{1u, 1u}, 8u, {10u, 2u}, &contexts[1], compile},
        {{2u, 1u}, 1u, {11u, 1u}, &contexts[2], compile},
    };
    fragment::fragment_compiler_registry_v1 registry{entries, 3u};
    assert(fragment::validate_fragment_compiler_registry_v1(registry));
    const auto *found = fragment::find_fragment_compiler_v1(
        registry, {1u, 1u}, 8u);
    assert(found == &entries[1]);
    program::prepared_program_v2 output{};
    joint::atom_fragment_request_v1 request{};
    assert(found->compile(found->source_context, request, found->candidate_id,
        &output) == fragment::fragment_compile_status_code_v1::success);
    assert(output.flags == 20u);
    assert(fragment::find_fragment_compiler_v1(
        registry, {1u, 1u}, 9u) == nullptr);

    entries[1].candidate_id = 7u;
    const auto duplicate =
        fragment::validate_fragment_compiler_registry_v1(registry);
    assert(duplicate.code == fragment::
        fragment_compiler_registry_status_code_v1::
            duplicate_or_unordered_entry);
    assert(duplicate.index == 1u);

    entries[1].candidate_id = 8u;
    entries[2].compile = nullptr;
    const auto missing =
        fragment::validate_fragment_compiler_registry_v1(registry);
    assert(missing.code == fragment::
        fragment_compiler_registry_status_code_v1::missing_compile_function);
    assert(missing.index == 2u);
}
