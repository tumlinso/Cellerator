#include <Cellerator/compute/architecture/provider.hh>
#include <Cellerator/compute/operation/candidate_catalog_v2.hh>
#include <Cellerator/runtime/device_descriptor.hh>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace architecture = cellerator::compute::architecture;
namespace core = cellerator::compute::math::core;
namespace execution = cellerator::execution;
namespace runtime = cellerator::runtime;

namespace {

constexpr architecture::architecture_identity_v1 provider_identity{11u, 12u};

architecture::matrix_engine_capability_v1 valid_capability() {
    architecture::matrix_engine_capability_v1 result{};
    result.identity = {21u, 22u};
    result.provider_identity = provider_identity;
    result.vendor = architecture::architecture_vendor_v1::nvidia;
    result.architecture_class = static_cast<std::uint16_t>(
        runtime::device_architecture_class_v1::nvidia_volta);
    result.minimum_compute_major = 7u;
    result.maximum_compute_major = 7u;
    result.maximum_compute_minor = 99u;
    result.instruction_family =
        architecture::matrix_instruction_family_v1::nvidia_wmma;
    result.collective_scope = architecture::collective_scope_v1::warp;
    result.collective_threads = 32u;
    result.instruction_m = 16u;
    result.instruction_n = 16u;
    result.instruction_k = 16u;
    result.operand_a_type = execution::numeric_type::f16;
    result.operand_b_type = execution::numeric_type::f16;
    result.accumulation_type = execution::numeric_type::f32;
    result.output_type = execution::numeric_type::f32;
    result.operand_a_layout = architecture::matrix_layout_v1::row_major;
    result.operand_b_layout = architecture::matrix_layout_v1::column_major;
    result.accumulation_layout = architecture::matrix_layout_v1::opaque;
    result.output_layout = architecture::matrix_layout_v1::row_major;
    result.instruction_sparsity = architecture::instruction_sparsity_v1::dense;
    result.flags = architecture::capability_source_linked_implementation
        | architecture::capability_fragment_layout_opaque
        | architecture::capability_requires_converged_collective;
    result.engine_requirements = architecture::matrix_engine_multiply_accumulate;
    return result;
}

architecture::architecture_provider_v1 valid_provider() {
    static const architecture::matrix_engine_capability_v1 capability =
        valid_capability();
    architecture::architecture_provider_v1 result{};
    result.identity = provider_identity;
    result.name = "validation-volta-provider";
    result.capabilities = &capability;
    result.capability_count = 1u;
    return result;
}

architecture::provider_status_v1 register_valid_provider(
    architecture::architecture_provider_registry_v1 *registry) noexcept {
    return architecture::register_architecture_provider_v1(
        registry, valid_provider());
}

architecture::provider_status_v1 reject_manifest_entry(
    architecture::architecture_provider_registry_v1 *) noexcept {
    return architecture::provider_status_v1::registration_failed;
}

runtime::device_descriptor_v1 valid_device() {
    runtime::device_descriptor_v1 result{};
    result.vendor = runtime::nvidia_pci_vendor_id;
    result.ordinal = 0;
    result.compute_major = 7u;
    result.architecture = runtime::device_architecture_class_v1::nvidia_volta;
    result.multiprocessor_count = 80u;
    result.warp_size = 32u;
    result.maximum_threads_per_block = 1024u;
    result.maximum_threads_per_multiprocessor = 2048u;
    result.maximum_blocks_per_multiprocessor = 32u;
    result.maximum_thread_dimensions[0] = 1024u;
    result.maximum_thread_dimensions[1] = 1024u;
    result.maximum_thread_dimensions[2] = 64u;
    result.maximum_grid_dimensions[0] = 2147483647u;
    result.maximum_grid_dimensions[1] = 65535u;
    result.maximum_grid_dimensions[2] = 65535u;
    result.registers_per_block = 65536u;
    result.registers_per_multiprocessor = 65536u;
    result.shared_memory_per_block_bytes = 49152u;
    result.optin_shared_memory_per_block_bytes = 98304u;
    result.shared_memory_per_multiprocessor_bytes = 98304u;
    result.global_memory_bytes = 16ull << 30u;
    result.l2_cache_bytes = 6ull << 20u;
    result.hardware_compatibility_identity = 1u;
    result.performance_class_identity = 2u;
    return result;
}

bool supports_fp32(const core::numeric_policy &numeric) noexcept {
    return numeric.sparse_storage == execution::numeric_type::f32
        && numeric.dense_storage == execution::numeric_type::f32
        && numeric.output_storage == execution::numeric_type::f32
        && numeric.multiply == execution::numeric_type::f32
        && numeric.accumulation == execution::numeric_type::f32
        && numeric.scalar == execution::numeric_type::f32;
}

std::uint32_t prepare_calls = 0u;

core::operation_status prepare_stub(
    const core::operation_candidate &,
    const core::operation_problem &,
    const core::structure_set_key &,
    const core::projection_key &,
    const core::numeric_policy &,
    const core::prepare_policy &,
    core::prepared_operation *) noexcept {
    ++prepare_calls;
    return {};
}

core::operation_candidate valid_candidate() {
    core::operation_candidate result{};
    result.identity = {31u, 32u};
    result.name = "validation-native-candidate";
    result.operation = core::operation_kind::weighted_relation_reduce;
    result.projection = core::projection_kind::native_row_masked;
    result.backend = core::backend_kind::native_direct;
    result.supports_numeric = supports_fp32;
    result.prepare = prepare_stub;
    return result;
}

} // namespace

int main() {
    architecture::architecture_provider_registry_v1 provider_registry{};
    const auto empty_provider_registry = provider_registry;

    architecture::architecture_provider_v1 invalid_provider = valid_provider();
    invalid_provider.identity = {};
    assert(architecture::register_architecture_provider_v1(
            &provider_registry, invalid_provider)
        == architecture::provider_status_v1::invalid_identity);
    assert(std::memcmp(&provider_registry, &empty_provider_registry,
               sizeof(provider_registry)) == 0);

    architecture::matrix_engine_capability_v1 stale_capability =
        valid_capability();
    stale_capability.provider_identity = {91u, 92u};
    architecture::architecture_provider_v1 stale_provider = valid_provider();
    stale_provider.capabilities = &stale_capability;
    assert(architecture::register_architecture_provider_v1(
            &provider_registry, stale_provider)
        == architecture::provider_status_v1::invalid_capability);
    assert(std::memcmp(&provider_registry, &empty_provider_registry,
               sizeof(provider_registry)) == 0);

    architecture::matrix_engine_capability_v1 incompatible_capability =
        valid_capability();
    incompatible_capability.output_type = execution::numeric_type::invalid;
    assert(architecture::validate_matrix_engine_capability_v1(
            incompatible_capability)
        == architecture::capability_status_v1::invalid_numeric_contract);
    architecture::architecture_provider_v1 incompatible_provider =
        valid_provider();
    incompatible_provider.capabilities = &incompatible_capability;
    assert(architecture::register_architecture_provider_v1(
            &provider_registry, incompatible_provider)
        == architecture::provider_status_v1::invalid_capability);

    const architecture::provider_registration_function_v1 registrations[]{
        register_valid_provider, reject_manifest_entry};
    assert(architecture::register_compiled_providers_v1(
            &provider_registry, {registrations, 2u})
        == architecture::provider_status_v1::registration_failed);
    assert(std::memcmp(&provider_registry, &empty_provider_registry,
               sizeof(provider_registry)) == 0);

    assert(register_valid_provider(&provider_registry)
        == architecture::provider_status_v1::success);
    const auto one_provider_registry = provider_registry;
    assert(register_valid_provider(&provider_registry)
        == architecture::provider_status_v1::duplicate_provider);
    assert(std::memcmp(&provider_registry, &one_provider_registry,
               sizeof(provider_registry)) == 0);
    assert(architecture::seal_architecture_provider_registry_v1(
            &provider_registry)
        == architecture::provider_status_v1::success);
    assert(register_valid_provider(&provider_registry)
        == architecture::provider_status_v1::invalid_argument);

    runtime::device_descriptor_v1 wrong_device = valid_device();
    wrong_device.compute_major = 8u;
    wrong_device.architecture =
        runtime::device_architecture_class_v1::nvidia_ampere;
    const architecture::architecture_provider_v1 *active[1]{
        reinterpret_cast<const architecture::architecture_provider_v1 *>(1)};
    std::uint32_t active_count = 99u;
    assert(architecture::active_architecture_providers_v1(
            provider_registry, wrong_device, active, 1u, &active_count)
        == architecture::provider_status_v1::success);
    assert(active_count == 0u);

    runtime::device_descriptor_v1 sealed_query_output{};
    const runtime::device_descriptor_v1 query_baseline = sealed_query_output;
    std::uint64_t query_count = 0u;
    assert(runtime::query_device_descriptor_v1(
            0, true, &sealed_query_output, &query_count)
        == runtime::device_descriptor_status_v1::invalid_state);
    assert(query_count == 0u);
    assert(std::memcmp(&sealed_query_output, &query_baseline,
               sizeof(sealed_query_output)) == 0);

    core::candidate_registry candidate_registry{};
    const auto empty_candidate_registry = candidate_registry;
    core::operation_candidate invalid_candidate = valid_candidate();
    invalid_candidate.prepare = nullptr;
    assert(core::register_candidate(&candidate_registry, invalid_candidate).code
        == core::operation_status_code::invalid_argument);
    assert(std::memcmp(&candidate_registry, &empty_candidate_registry,
               sizeof(candidate_registry)) == 0);

    const core::operation_candidate candidate = valid_candidate();
    assert(core::register_candidate(&candidate_registry, candidate));
    const auto one_candidate_registry = candidate_registry;
    assert(core::register_candidate(&candidate_registry, candidate).code
        == core::operation_status_code::duplicate_candidate);
    assert(std::memcmp(&candidate_registry, &one_candidate_registry,
               sizeof(candidate_registry)) == 0);

    core::candidate_descriptor_v2 invalid_descriptor{};
    invalid_descriptor.candidate = candidate;
    invalid_descriptor.provider_identity = {41u, 42u};
    invalid_descriptor.projection_contract = {{51u, 52u}, 1u, 0u, 1u, 0u};
    invalid_descriptor.flags = core::candidate_descriptor_requires_capability;
    assert(core::validate_candidate_descriptor_v2(invalid_descriptor)
        == core::candidate_catalog_status_v2::invalid_candidate);

    const core::operation_problem problem{core::operation_core_schema_version,
        core::operation_kind::weighted_relation_reduce, 0u, {61u, 62u},
        1u, 1u, 64u};
    core::structure_set_key structures{};
    structures.count = 1u;
    structures.structures[0] = {{71u, 72u}, {73u, 1u}, {1u}};
    const core::projection_key projection{{81u, 82u}, {83u, 1u},
        core::projection_kind::native_row_masked, 1u, 0u};
    core::numeric_policy numeric{};
    numeric.sparse_storage = execution::numeric_type::f32;
    numeric.dense_storage = execution::numeric_type::f16;
    numeric.output_storage = execution::numeric_type::f32;
    numeric.multiply = execution::numeric_type::f32;
    numeric.accumulation = execution::numeric_type::f32;
    numeric.scalar = execution::numeric_type::f32;
    numeric.bias = execution::numeric_type::f32;
    core::prepared_operation prepared{};
    assert(core::prepare_candidate(candidate, problem, structures, projection,
               numeric, {}, &prepared).code
        == core::operation_status_code::unsupported_numeric_policy);
    assert(prepare_calls == 0u);

    std::cout << "foundation_negative_test passed providers=1 candidates=1"
              << " atomic_rollback=1 sealed_queries=" << query_count << '\n';
    return 0;
}
