#include "Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/apply_registration_v1.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {
namespace {

bool valid_bindings(apply_kernel_binding_view_v1 bindings) noexcept {
    if (bindings.binding_count != 0u && bindings.bindings == nullptr) {
        return false;
    }
    std::uint64_t previous = 0u;
    for (std::uint64_t index = 0u; index < bindings.binding_count; ++index) {
        const apply_kernel_binding_v1 &binding = bindings.bindings[index];
        if (binding.candidate_id == 0u || binding.candidate_id <= previous
            || binding.kernel_symbol == nullptr) {
            return false;
        }
        previous = binding.candidate_id;
    }
    return true;
}

}  // namespace

apply_registration_status_v1 register_sm70_apply_candidates_v1(
    const sm70_apply_inventory_v1 &inventory,
    apply_kernel_binding_view_v1 bindings,
    bool query_attributes,
    apply_registration_workspace_v1 workspace) noexcept {
    if (validate_sm70_apply_inventory_v1(inventory)
        != apply_inventory_status_v1::success) {
        return apply_registration_status_v1::invalid_inventory;
    }
    if (!valid_bindings(bindings)) {
        return apply_registration_status_v1::invalid_bindings;
    }
    if (workspace.registrations == nullptr || workspace.receipts == nullptr
        || workspace.registration_capacity < inventory.candidate_count
        || workspace.receipt_capacity < inventory.candidate_count) {
        return apply_registration_status_v1::insufficient_capacity;
    }
    std::uint64_t binding_index = 0u;
    bool query_failure = false;
    for (std::uint64_t index = 0u; index < inventory.candidate_count; ++index) {
        const catalog_v3::candidate_descriptor_v3 &candidate =
            inventory.candidates[index];
        while (binding_index < bindings.binding_count
            && bindings.bindings[binding_index].candidate_id
                < candidate.identity.candidate_id) {
            ++binding_index;
        }
        const void *symbol = nullptr;
        if (binding_index < bindings.binding_count
            && bindings.bindings[binding_index].candidate_id
                == candidate.identity.candidate_id) {
            symbol = bindings.bindings[binding_index].kernel_symbol;
        }
        workspace.registrations[index] = {&candidate,
            &inventory.capabilities[index], symbol};
        apply_resource_receipt_v1 receipt{};
        receipt.candidate_id = candidate.identity.candidate_id;
        receipt.stage_id = candidate.stages[0].stage_id;
        receipt.threads_per_cta = candidate.resources.threads_per_cta;
        receipt.static_shared_bytes = candidate.resources.shared_bytes_per_cta;
        if (symbol == nullptr) {
            receipt.state = query_attributes
                ? apply_resource_receipt_state_v1::compiled_symbol_unavailable
                : apply_resource_receipt_state_v1::declared_only;
        } else if (!query_attributes) {
            receipt.state = apply_resource_receipt_state_v1::declared_only;
        } else {
            cudaFuncAttributes attributes{};
            const cudaError_t error = cudaFuncGetAttributes(&attributes, symbol);
            receipt.cuda_error = static_cast<std::int32_t>(error);
            if (error == cudaSuccess) {
                const bool attributes_fit = attributes.sharedSizeBytes
                        <= std::numeric_limits<std::uint32_t>::max()
                    && attributes.numRegs >= 0
                    && attributes.maxThreadsPerBlock >= 0
                    && attributes.ptxVersion >= 0
                    && attributes.binaryVersion >= 0;
                if (attributes_fit) {
                    receipt.state = apply_resource_receipt_state_v1::
                        compiled_query_complete;
                    receipt.static_shared_bytes =
                        static_cast<std::uint32_t>(attributes.sharedSizeBytes);
                    receipt.registers_per_thread =
                        static_cast<std::uint32_t>(attributes.numRegs);
                    receipt.maximum_threads_per_block =
                        static_cast<std::uint32_t>(attributes.maxThreadsPerBlock);
                    receipt.ptx_version = static_cast<std::uint32_t>(
                        attributes.ptxVersion);
                    receipt.binary_version = static_cast<std::uint32_t>(
                        attributes.binaryVersion);
                } else {
                    receipt.state =
                        apply_resource_receipt_state_v1::cuda_query_failed;
                    receipt.cuda_error = -1;
                    query_failure = true;
                }
            } else {
                receipt.state =
                    apply_resource_receipt_state_v1::cuda_query_failed;
                query_failure = true;
            }
        }
        workspace.receipts[index] = receipt;
    }
    return query_failure ? apply_registration_status_v1::cuda_query_failure
                         : apply_registration_status_v1::success;
}

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
