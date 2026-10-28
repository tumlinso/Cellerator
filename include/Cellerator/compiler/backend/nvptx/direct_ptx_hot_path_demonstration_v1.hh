#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::backend::nvptx {

struct unit_degree_relation_ptx_request_v1 {
    std::uint32_t row_count = 0u;
    std::uint32_t dense_input_count = 0u;
    std::uint16_t target_sm_major = 0u;
    std::uint16_t target_sm_minor = 0u;
};

enum class unit_degree_relation_ptx_status_v1 : std::uint8_t {
    success = 0u,
    invalid_shape,
    unsupported_target,
};

struct unit_degree_relation_ptx_result_v1 {
    unit_degree_relation_ptx_status_v1 status =
        unit_degree_relation_ptx_status_v1::invalid_shape;
    std::string kernel_symbol;
    std::string ptx;
    std::vector<std::string> restrictions;

    explicit operator bool() const noexcept {
        return status == unit_degree_relation_ptx_status_v1::success;
    }
};

// Lowers the exact one-edge-per-row relation operation
// output[row] = weight[row] * input[column[row]] directly to SM70 PTX.
[[nodiscard]] unit_degree_relation_ptx_result_v1
lower_unit_degree_relation_apply_directly_to_ptx_v1(
    const unit_degree_relation_ptx_request_v1& request);

}  // namespace Cellerator::compiler::backend::nvptx
