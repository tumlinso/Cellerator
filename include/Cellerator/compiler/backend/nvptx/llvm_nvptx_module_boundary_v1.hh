#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::backend::nvptx {

enum class nvptx_scalar_operation_v1 : std::uint8_t {
    add_f32 = 1u,
};

struct nvptx_realization_operation_v1 {
    std::uint64_t realization_node_identity = 0u;
    nvptx_scalar_operation_v1 operation = nvptx_scalar_operation_v1::add_f32;
    std::string symbol;
};

struct nvptx_module_request_v1 {
    std::uint64_t realization_module_identity = 0u;
    std::uint32_t compute_major = 0u;
    std::uint32_t compute_minor = 0u;
    std::vector<nvptx_realization_operation_v1> operations;
};

enum class nvptx_module_status_v1 : std::uint8_t {
    success = 0u,
    invalid_module,
    unsupported_operation,
};

struct nvptx_module_result_v1 {
    nvptx_module_status_v1 status = nvptx_module_status_v1::invalid_module;
    std::string llvm_ir;

    explicit operator bool() const noexcept {
        return status == nvptx_module_status_v1::success;
    }
};

// LLVM implementation types remain private to the backend. The stable public
// boundary accepts Cellerator-owned records and returns portable textual IR.
[[nodiscard]] nvptx_module_result_v1 lower_realization_subset_to_llvm_nvptx_v1(
    const nvptx_module_request_v1& request);

}  // namespace Cellerator::compiler::backend::nvptx
