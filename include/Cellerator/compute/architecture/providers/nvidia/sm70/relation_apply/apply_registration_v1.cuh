#pragma once

#include <Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/apply_inventory_v1.hh>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {

enum class apply_resource_receipt_state_v1 : std::uint8_t {
    declared_only = 1u,
    compiled_query_complete = 2u,
    compiled_symbol_unavailable = 3u,
    cuda_query_failed = 4u,
};

struct apply_kernel_binding_v1 {
    std::uint64_t candidate_id = 0u;
    const void *kernel_symbol = nullptr;
};

struct apply_kernel_binding_view_v1 {
    const apply_kernel_binding_v1 *bindings = nullptr;
    std::uint64_t binding_count = 0u;
};

struct apply_candidate_registration_v1 {
    const catalog_v3::candidate_descriptor_v3 *candidate = nullptr;
    const apply_candidate_capability_v1 *capability = nullptr;
    const void *kernel_symbol = nullptr;
};

struct apply_resource_receipt_v1 {
    std::uint64_t candidate_id = 0u;
    std::uint64_t stage_id = 0u;
    apply_resource_receipt_state_v1 state =
        apply_resource_receipt_state_v1::declared_only;
    std::uint8_t reserved[3]{};
    std::uint32_t threads_per_cta = 0u;
    std::uint32_t static_shared_bytes = 0u;
    std::uint32_t registers_per_thread = 0u;
    std::uint32_t maximum_threads_per_block = 0u;
    std::uint32_t ptx_version = 0u;
    std::uint32_t binary_version = 0u;
    std::int32_t cuda_error = 0;
    std::uint32_t reserved1 = 0u;
};

struct apply_registration_workspace_v1 {
    apply_candidate_registration_v1 *registrations = nullptr;
    std::uint64_t registration_capacity = 0u;
    apply_resource_receipt_v1 *receipts = nullptr;
    std::uint64_t receipt_capacity = 0u;
};

enum class apply_registration_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_inventory,
    invalid_bindings,
    insufficient_capacity,
    cuda_query_failure,
};

// Bindings are sorted by candidate ID and may be sparse. query_attributes is a
// cold opt-in; false produces deterministic declared receipts without touching
// CUDA runtime state. Missing symbols remain explicitly visible.
apply_registration_status_v1 register_sm70_apply_candidates_v1(
    const sm70_apply_inventory_v1 &inventory,
    apply_kernel_binding_view_v1 bindings,
    bool query_attributes,
    apply_registration_workspace_v1 workspace) noexcept;

static_assert(std::is_trivially_copyable<apply_candidate_registration_v1>::value,
    "candidate registrations must remain pointer-first cold records");
static_assert(std::is_trivially_copyable<apply_resource_receipt_v1>::value,
    "resource receipts must remain stable POD evidence");

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
