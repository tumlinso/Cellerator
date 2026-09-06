#pragma once

#include <cstdint>

namespace cellerator::compute::operation {

// The historical contract_on_support kind alone cannot determine result shape.
enum class support_product_result : std::uint8_t {
    unspecified = 0u,
    scalar_dot = 1u,          // dot[e] = sum_k A[s(e),k] * B[d(e),k]
    edge_channel_product = 2u // product[e,k] = A[s(e),k] * B[d(e),k]
};

enum class support_product_assembly : std::uint8_t {
    unspecified = 0u,
    sum_partials = 1u,
    concatenate_channels = 2u
};

constexpr bool valid_support_product_assembly(support_product_result result,
    support_product_assembly assembly) noexcept {
    return (result == support_product_result::scalar_dot
            && assembly == support_product_assembly::sum_partials)
        || (result == support_product_result::edge_channel_product
            && assembly == support_product_assembly::concatenate_channels);
}

constexpr std::uint64_t support_product_output_width(
    support_product_result result, std::uint64_t embedding_width) noexcept {
    return result == support_product_result::scalar_dot ? 1u
        : result == support_product_result::edge_channel_product
            ? embedding_width : 0u;
}

} // namespace cellerator::compute::operation
