#include <Cellerator/compute/operation/model_ops/model_ops.hh>

#include <cstdint>
#include <limits>

int main() {
    using namespace cellerator::compute::operation::model_ops;
    constexpr auto too_many = maximum_indexed_count + 1;

    const auto oversized_pairs = dense_reduce_pair_forward(
        nullptr, nullptr, nullptr, nullptr, too_many, 0, 0,
        0.0f, 0.0f, 0.0f, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr);
    if (oversized_pairs != cudaErrorInvalidValue) return 1;

    const auto byte_wrapped_latent_extent = dense_reduce_pair_forward(
        nullptr, nullptr, nullptr, nullptr, 0, 1, std::int64_t{1} << 62,
        0.0f, 0.0f, 0.0f, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr);
    if (byte_wrapped_latent_extent != cudaErrorInvalidValue) return 2;

    const auto oversized_bucket_rows = developmental_stage_bucket_forward(
        nullptr, nullptr, too_many, 0, 0.0f, 0.0f, false, 0,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr);
    if (oversized_bucket_rows != cudaErrorInvalidValue) return 3;

    const auto grid_overflow = weighted_future_target(
        nullptr, nullptr, nullptr, 0, 2147483647LL * 256LL + 1LL, 0, 1,
        nullptr, nullptr);
    if (grid_overflow != cudaErrorInvalidValue) return 4;

    const auto product_overflow = weighted_future_target(
        nullptr, nullptr, nullptr, 0, std::numeric_limits<std::int64_t>::max(), 0, 2,
        nullptr, nullptr);
    if (product_overflow != cudaErrorInvalidValue) return 5;

    const auto neighbor_product_overflow = weighted_future_target(
        nullptr, nullptr, nullptr, 0, 2, std::numeric_limits<std::int64_t>::max(), 0,
        nullptr, nullptr);
    if (neighbor_product_overflow != cudaErrorInvalidValue) return 6;

    const auto reference_extent_overflow = weighted_future_target(
        nullptr, nullptr, nullptr, 2, 1, 0, std::numeric_limits<std::int64_t>::max(),
        nullptr, nullptr);
    if (reference_extent_overflow != cudaErrorInvalidValue) return 7;
    return 0;
}
