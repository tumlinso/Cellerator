#pragma once

#include <Cellerator/compute/operation/builtin_catalog.hh>
#include <Cellerator/compute/candidate/csr_fallback_candidate.hh>
#include <Cellerator/compute/candidate/feature_major_small_n_candidate.hh>
#include <Cellerator/compute/candidate/row_masked_n1_candidate.hh>
#include <Cellerator/compute/candidate/transpose_backward_candidate.hh>
#include <Cellerator/execution/projection_activation_v2.hh>
#include <Cellerator/runtime/session.cuh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::compute::math::core {

inline constexpr std::uint32_t preparation_factory_schema_version = 1u;
inline constexpr std::uint32_t candidate_preparation_request_schema_version_v2 =
    2u;
inline constexpr std::uint32_t candidate_preparation_adapter_schema_version_v2 =
    2u;

// Host state storage is owned by the session owner and registered in the sole
// execution session's plan cache after successful preparation. The factory
// never allocates it and never owns projection or value bytes.
struct preparation_state_storage {
    void *data = nullptr;
    std::size_t bytes = 0u;
};

// Provider-independent inputs shared by all v2 preparation adapters. The
// activated projection reference supplies the projection key and erased view;
// only the adapter owned by the selected catalog entry interprets its bytes.
struct candidate_preparation_request_v2 {
    std::uint32_t schema_version =
        candidate_preparation_request_schema_version_v2;
    std::uint32_t reserved = 0u;
    operation_problem problem{};
    structure_set_key structures{};
    numeric_policy numeric{};
    prepare_policy policy{};
    runtime::execution_session *session = nullptr;
    std::uint32_t dense_width = 0u;
    std::uint32_t reserved2 = 0u;
    execution::axis_identity feature_axis{};
    execution::axis_identity row_axis{};
    execution::axis_identity dense_column_axis{};
    preparation_state_storage state{};
};

struct candidate_preparation_adapter_v2;

using erased_candidate_prepare_function_v2 = operation_status (*)(
    const candidate_preparation_adapter_v2 &,
    const candidate_preparation_request_v2 &,
    const execution::activated_projection_reference_v2 &,
    prepared_operation *) noexcept;

// This is the preparation-bearing catalog entry consumed by program v2.
// Providers publish one immutable record beside each descriptor. Adding a new
// projection therefore adds an adapter callback, not a central type switch.
struct candidate_preparation_adapter_v2 {
    std::uint32_t schema_version =
        candidate_preparation_adapter_schema_version_v2;
    std::uint32_t record_bytes = sizeof(candidate_preparation_adapter_v2);
    const candidate_descriptor_v2 *candidate = nullptr;
    erased_candidate_prepare_function_v2 prepare = nullptr;
    std::uint64_t reserved[2]{};
};

struct candidate_preparation_catalog_v2 {
    const candidate_preparation_adapter_v2 *entries = nullptr;
    std::uint32_t entry_count = 0u;
    std::uint32_t reserved = 0u;
};

operation_status validate_candidate_preparation_adapter_v2(
    const candidate_preparation_adapter_v2 &adapter) noexcept;

const candidate_preparation_adapter_v2 *find_candidate_preparation_adapter_v2(
    candidate_preparation_catalog_v2 catalog,
    stable_id candidate_identity) noexcept;

// Compatibility adapters for the current five-entry fragment. Their callback
// and descriptor pointers are stable for process lifetime; no registration,
// allocation, or physical-format discrimination occurs here.
candidate_preparation_catalog_v2
built_in_candidate_preparation_catalog_v2() noexcept;

operation_status prepare_catalog_candidate_v2(
    const candidate_preparation_adapter_v2 &adapter,
    const candidate_preparation_request_v2 &request,
    const execution::activated_projection_reference_v2 &projection,
    prepared_operation *prepared) noexcept;

struct preparation_factory_request {
    std::uint32_t schema_version = preparation_factory_schema_version;
    const built_in_candidate_descriptor *catalog_entry = nullptr;
    operation_problem problem{};
    structure_set_key structures{};
    projection_key projection{};
    numeric_policy numeric{};
    prepare_policy policy{};
    runtime::execution_session *session = nullptr;
    std::uint32_t dense_width = 0u;
    execution::axis_identity feature_axis{};
    execution::axis_identity row_axis{};
    execution::axis_identity dense_column_axis{};
    preparation_state_storage state{};
};

// These typed overloads are the complete factory surface. They deliberately do
// not introduce a global projection variant, virtual dispatch, or a second
// candidate registry.
operation_status prepare_catalog_row_masked(
    const preparation_factory_request &request,
    const cellpack::persistent_packing_payload_view &projection,
    prepared_operation *prepared) noexcept;

operation_status prepare_catalog_csr(
    const preparation_factory_request &request,
    const execution_csr_view &projection,
    prepared_operation *prepared) noexcept;

operation_status prepare_catalog_feature_major(
    const preparation_factory_request &request,
    const feature_major_projection_view &projection,
    prepared_operation *prepared) noexcept;

operation_status prepare_catalog_transpose(
    const preparation_factory_request &request,
    const transpose_projection_view &projection,
    prepared_operation *prepared) noexcept;

static_assert(
    std::is_trivially_copyable<candidate_preparation_request_v2>::value,
    "erased preparation requests must remain pointer-copyable");
static_assert(
    std::is_trivially_copyable<candidate_preparation_adapter_v2>::value,
    "candidate-owned preparation adapters must remain pointer-copyable");
static_assert(
    std::is_standard_layout<candidate_preparation_adapter_v2>::value,
    "candidate-owned preparation adapters must remain field-addressable");
static_assert(
    std::is_trivially_copyable<candidate_preparation_catalog_v2>::value,
    "preparation catalog views must remain pointer-copyable");

} // namespace cellerator::compute::math::core
