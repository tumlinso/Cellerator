#pragma once

#include <Cellerator/compute/math/operation_core/builtin_catalog.hh>
#include <Cellerator/compute/math/operation_core/csr_fallback_candidate.hh>
#include <Cellerator/compute/math/operation_core/feature_major_small_n_candidate.hh>
#include <Cellerator/compute/math/operation_core/row_masked_n1_candidate.hh>
#include <Cellerator/compute/math/operation_core/transpose_backward_candidate.hh>
#include <Cellerator/runtime/session.cuh>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::math::core {

inline constexpr std::uint32_t preparation_factory_schema_version = 1u;

// Host state storage is owned by the session owner and registered in the sole
// execution session's plan cache after successful preparation. The factory
// never allocates it and never owns projection or value bytes.
struct preparation_state_storage {
    void *data = nullptr;
    std::size_t bytes = 0u;
};

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

} // namespace cellerator::compute::math::core
