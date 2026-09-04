#include "mock_backend_provider_v1.hh"

#include <algorithm>

namespace cb = cellerator::compiler::backend::v1;
namespace {

constexpr char k_triple[] = "mock-cpu-unknown";

cb::backend_status_v1 discover(
    void*, cb::backend_target_v1* targets, std::size_t capacity,
    std::size_t* count, cb::backend_diagnostic_sink_v1) noexcept {
    if (count == nullptr) return cb::backend_status_v1::invalid_argument;
    *count = 1;
    if (capacity == 0) return cb::backend_status_v1::success;
    if (targets == nullptr) return cb::backend_status_v1::invalid_argument;
    targets[0].triple = {k_triple, sizeof(k_triple) - 1};
    return cb::backend_status_v1::success;
}

cb::backend_status_v1 capabilities(
    void*, cb::backend_target_v1, std::uint64_t* result,
    cb::backend_diagnostic_sink_v1) noexcept {
    if (result == nullptr) return cb::backend_status_v1::invalid_argument;
    *result = cb::backend_capability_ordinary_object_v1;
    return cb::backend_status_v1::success;
}

cb::backend_status_v1 admissible(
    void*, cb::backend_target_v1, cb::backend_realization_view_v1 realization,
    cb::backend_diagnostic_sink_v1) noexcept {
    return realization.schema_version == 1 ? cb::backend_status_v1::success
                                           : cb::backend_status_v1::inadmissible_realization;
}

cb::backend_status_v1 emit(
    void*, cb::backend_target_v1, cb::backend_realization_view_v1 realization,
    cb::backend_object_buffer_v1* object, cb::backend_diagnostic_sink_v1) noexcept {
    if (object->capacity < realization.size) {
        object->size = realization.size;
        return cb::backend_status_v1::insufficient_capacity;
    }
    std::copy_n(realization.data, realization.size, object->data);
    object->size = realization.size;
    return cb::backend_status_v1::success;
}

}  // namespace

cb::backend_provider_v1 make_mock_backend_provider_v1() noexcept {
    return {
        cb::backend_provider_abi_version_v1,
        sizeof(cb::backend_provider_v1),
        nullptr,
        {{"mock", 4}, {"mock-cxx", 8}, {"1", 1}, {"mock-build", 10}},
        discover,
        capabilities,
        admissible,
        emit,
        nullptr,
    };
}
