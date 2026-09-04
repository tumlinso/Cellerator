#pragma once

#include <cstdint>

namespace cellerator::compiler::backend::nvcc::v1 {

inline constexpr std::uint32_t nvcc_backend_contract_version = 1;
inline constexpr std::uint32_t required_backend_abi_version = 1;
inline constexpr std::uint32_t required_realization_ir_version = 1;

enum class nvcc_backend_receipt_kind : std::uint32_t {
    source_emission = 1U << 0U,
    provider_binding = 1U << 1U,
    action_graph = 1U << 2U,
    ordinary_object = 1U << 3U,
    fatbinary = 1U << 4U,
    diagnostics = 1U << 5U,
    asynchronous_readiness = 1U << 6U,
    complete_cost = 1U << 7U,
};

inline constexpr std::uint32_t all_nvcc_backend_receipts =
    static_cast<std::uint32_t>(nvcc_backend_receipt_kind::source_emission) |
    static_cast<std::uint32_t>(nvcc_backend_receipt_kind::provider_binding) |
    static_cast<std::uint32_t>(nvcc_backend_receipt_kind::action_graph) |
    static_cast<std::uint32_t>(nvcc_backend_receipt_kind::ordinary_object) |
    static_cast<std::uint32_t>(nvcc_backend_receipt_kind::fatbinary) |
    static_cast<std::uint32_t>(nvcc_backend_receipt_kind::diagnostics) |
    static_cast<std::uint32_t>(nvcc_backend_receipt_kind::asynchronous_readiness) |
    static_cast<std::uint32_t>(nvcc_backend_receipt_kind::complete_cost);

// Pointer-free publication receipt for the compiler-owned NVCC backend.
// Hash halves identify the source-linked validated object without embedding
// cold provenance in a hot launch ABI.
struct nvcc_backend_receipt {
    std::uint32_t contract_version = nvcc_backend_contract_version;
    std::uint32_t backend_abi_version = required_backend_abi_version;
    std::uint32_t realization_ir_version = required_realization_ir_version;
    std::uint32_t receipt_mask = 0;
    std::uint64_t source_revision_high = 0;
    std::uint64_t source_revision_low = 0;
    std::uint64_t object_hash_high = 0;
    std::uint64_t object_hash_low = 0;
    std::uint32_t compute_capability = 0;
    bool exact_output = false;
    bool conventional_fallback_retained = false;
};

} // namespace cellerator::compiler::backend::nvcc::v1
