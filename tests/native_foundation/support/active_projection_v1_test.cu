#include <Cellerator/compute/operation/active_projection/active_projection_v1.cuh>
#include <Cellerator/compute/operation/edge/dynamic_support_mask_v1.cuh>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace projection = cellerator::compute::operation::active_projection;
namespace edge = cellerator::compute::operation::edge;

namespace {

template <class T> struct device_buffer {
    T *data = nullptr;
    explicit device_buffer(std::size_t count) { assert(cudaMalloc(&data, count * sizeof(T)) == cudaSuccess); }
    ~device_buffer() { cudaFree(data); }
    device_buffer(const device_buffer &) = delete;
};

projection::projection_build_request_v1 make_request(
    const std::uint8_t *primal_active, const std::uint8_t *derivative_active,
    std::uint32_t *primal_indices, std::uint32_t *derivative_indices,
    std::uint64_t epoch, std::uint64_t generation) {
    return {{71u, epoch, {projection::activity_scope_v1::per_instance, 9u,
                 generation}}, primal_active, derivative_active, 6u,
        primal_indices, 6u, derivative_indices, 6u};
}

void require_cuda(cudaError_t status) { assert(status == cudaSuccess); }

void exact_compaction_matches_persistent_mask_and_excludes_inactive_nan() {
    const float host_input[]{2.0f, std::numeric_limits<float>::quiet_NaN(),
        -3.0f, 11.0f, std::numeric_limits<float>::quiet_NaN(), 5.0f};
    const std::uint8_t primal_active[]{1u, 0u, 1u, 0u, 0u, 1u};
    const std::uint8_t derivative_active[]{1u, 1u, 1u, 0u, 1u, 1u};
    std::uint32_t primal_indices[6]{};
    std::uint32_t derivative_indices[6]{};
    projection::projection_map_v1 primal{};
    projection::projection_map_v1 derivative{};
    assert(projection::build_exact_projection_v1(make_request(primal_active,
        derivative_active, primal_indices, derivative_indices, 3u, 8u),
        &primal, &derivative) == projection::status_v1::success);
    assert(primal.active_count == 3u && derivative.active_count == 5u);
    assert(primal_indices[0] == 0u && primal_indices[1] == 2u
        && primal_indices[2] == 5u);

    device_buffer<float> input(6u), compact(3u), scattered(6u), persistent(6u);
    device_buffer<std::uint8_t> mask(6u);
    device_buffer<std::uint32_t> map(3u);
    require_cuda(cudaMemcpy(input.data, host_input, sizeof(host_input), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(mask.data, primal_active, sizeof(primal_active), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(map.data, primal_indices, 3u * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    assert(projection::enqueue_compact_copy_v1({input.data, map.data, compact.data,
        primal.active_count}) == projection::status_v1::success);
    assert(projection::enqueue_compact_scatter_v1({compact.data, map.data,
        scattered.data, 6u, primal.active_count}) == projection::status_v1::success);
    assert(edge::enqueue_dynamic_support_mask_v1({{17u, 6u}, input.data,
        persistent.data, mask.data, edge::mask_encoding_v1::byte_per_edge,
        71u, 3u, 1u, 8u, 2u, 0u, nullptr}) == edge::status_v1::success);
    float compact_result[6]{};
    float mask_result[6]{};
    require_cuda(cudaMemcpy(compact_result, scattered.data, sizeof(compact_result), cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(mask_result, persistent.data, sizeof(mask_result), cudaMemcpyDeviceToHost));
    require_cuda(cudaDeviceSynchronize());
    for (std::uint32_t i = 0u; i < 6u; ++i) {
        if (primal_active[i] == 0u) {
            assert(compact_result[i] == 0.0f && mask_result[i] == 0.0f);
        } else {
            assert(compact_result[i] == mask_result[i]);
        }
    }
}

void activity_epoch_and_approximation_are_explicit() {
    const std::uint8_t primal_active[]{1u, 1u, 1u, 1u, 1u, 1u};
    const std::uint8_t derivative_active[]{1u, 1u, 1u, 1u, 1u, 1u};
    const float weights[]{0.0f, 0.0001f, 2.0f, -3.0f, 4.0f, 5.0f};
    std::uint32_t primal_indices[6]{};
    std::uint32_t derivative_indices[6]{};
    projection::projection_map_v1 primal{};
    projection::projection_map_v1 derivative{};
    const auto exact = make_request(primal_active, derivative_active,
        primal_indices, derivative_indices, 3u, 9u);
    assert(projection::build_exact_projection_v1(exact, &primal, &derivative)
        == projection::status_v1::success);
    assert(primal.identity.activity.generation == 9u
        && primal.identity.structure_epoch == 3u);
    const projection::approximate_drop_policy_v1 policy{true,
        projection::approximation_provenance_v1::unassessed, 0.001f, 0.0f, 0u};
    assert(projection::build_approximate_primal_projection_v1(
        {exact, weights, policy}, &primal, &derivative)
        == projection::status_v1::success);
    assert(primal.active_count == 4u && derivative.active_count == 6u);
    assert(projection::validate_approximate_drop_policy_v1(
        {true, projection::approximation_provenance_v1::bounded, 0.1f,
            -1.0f, 0u}) == projection::status_v1::invalid_argument);

    // Approximation changes only the primal admission map.  Restore the exact
    // primal map before exercising the zero-weight response identity.
    assert(projection::build_exact_projection_v1(exact, &primal, &derivative)
        == projection::status_v1::success);

    const float input[]{7.0f, 13.0f, 2.0f, 3.0f, 5.0f, 11.0f};
    float primal_result[6]{};
    float response_result[6]{};
    device_buffer<float> d_input(6u), d_weights(6u), d_primal(6u), d_response(6u);
    device_buffer<std::uint32_t> d_primal_indices(6u), d_derivative_indices(6u);
    require_cuda(cudaMemcpy(d_input.data, input, sizeof(input), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(d_weights.data, weights, sizeof(weights), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(d_primal_indices.data, primal_indices,
        6u * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(d_derivative_indices.data, derivative_indices,
        6u * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    require_cuda(cudaMemset(d_primal.data, 0, 6u * sizeof(float)));
    require_cuda(cudaMemset(d_response.data, 0, 6u * sizeof(float)));
    assert(projection::enqueue_weighted_response_v1({d_input.data, d_weights.data,
        d_primal.data, d_response.data, d_primal_indices.data, primal.active_count,
        d_derivative_indices.data, 6u}) == projection::status_v1::success);
    require_cuda(cudaMemcpy(primal_result, d_primal.data, sizeof(primal_result), cudaMemcpyDeviceToHost));
    require_cuda(cudaMemcpy(response_result, d_response.data, sizeof(response_result), cudaMemcpyDeviceToHost));
    require_cuda(cudaDeviceSynchronize());
    assert(primal_result[0] == 0.0f && response_result[0] == input[0]);
}

} // namespace

int main() {
    exact_compaction_matches_persistent_mask_and_excludes_inactive_nan();
    activity_epoch_and_approximation_are_explicit();
}
