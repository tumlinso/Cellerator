#pragma once

#include <Cellerator/execution/object_binding/multi_extent_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::candidate::jbc_multi_extent {

struct relation_apply_state_v1 {
    const float *extent_scales = nullptr;
    std::uint64_t extent_count = 0u;
    float bias = 0.0f;
};

execution::object_binding::binding_status_v1
launch_relation_apply_host_v1(const void *prepared_state,
    const execution::object_binding::multi_extent_physical_binding_v1 &input,
    void *output, std::uint64_t output_bytes,
    void *caller_stream) noexcept;

execution::object_binding::direct_multi_extent_candidate_v1
make_experimental_relation_apply_candidate_v1(
    const relation_apply_state_v1 *state,
    execution::object_binding::stable_identity_v1 candidate_identity) noexcept;

static_assert(std::is_trivially_copyable_v<relation_apply_state_v1>);

}  // namespace cellerator::compute::candidate::jbc_multi_extent
