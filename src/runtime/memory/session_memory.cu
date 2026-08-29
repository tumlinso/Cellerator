#include <Cellerator/memory/session_memory.cuh>

#include <cstdint>

namespace cellerator::memory {

namespace {

status from_session_status(runtime::session_status value) noexcept {
    switch (value) {
    case runtime::session_status::success: return status::success;
    case runtime::session_status::invalid_argument: return status::invalid_argument;
    case runtime::session_status::invalid_state: return status::invalid_state;
    case runtime::session_status::capacity_exceeded:
    case runtime::session_status::workspace_exhausted:
        return status::capacity_exceeded;
    case runtime::session_status::device_mismatch: return status::invalid_placement;
    case runtime::session_status::cuda_failure: return status::cuda_failure;
    }
    return status::cuda_failure;
}

bool session_device_placement(
    const runtime::execution_session &session,
    placement where) noexcept {
    return where.kind == domain::device
        && where.device_ordinal == session.device
        && where.numa_node == -1;
}

} // namespace

status reserve_session_allocation(
    runtime::execution_session *session,
    runtime::persistent_lifetime lifetime,
    const allocation_request &request,
    std::uint32_t generation,
    allocation *out) noexcept {
    if (out != nullptr) *out = allocation{};
    if (session == nullptr || out == nullptr || generation == 0u)
        return status::invalid_argument;
    const status request_status = validate_allocation_request(request);
    if (request_status != status::success) return request_status;
    if (!session_device_placement(*session, request.where))
        return status::invalid_placement;
    // cudaMalloc provides at least 256-byte alignment on supported CUDA
    // devices. Wider requirements need a separately prepared allocator.
    if (request.alignment > 256u) return status::invalid_alignment;
    void *base = nullptr;
    const runtime::session_status reserved = runtime::reserve_persistent(
        session, lifetime, request.bytes, &base);
    const status converted = from_session_status(reserved);
    if (converted != status::success) return converted;
    *out = allocation{
        base, request.bytes, request.alignment, request.where, generation};
    return status::success;
}

status reserve_session_workspace(
    runtime::execution_session *session,
    std::uint32_t stream_index,
    const workspace_requirement &requirement,
    workspace *out) noexcept {
    if (out != nullptr) *out = workspace{};
    if (session == nullptr || out == nullptr) return status::invalid_argument;
    if (!valid_alignment(requirement.alignment)) return status::invalid_alignment;
    if (!session_device_placement(*session, requirement.where))
        return status::invalid_placement;
    if (requirement.alignment > 256u) return status::invalid_alignment;
    void *base = nullptr;
    const status converted = from_session_status(runtime::reserve_transient(
        session, stream_index, requirement.bytes, &base));
    if (converted != status::success) return converted;
    *out = workspace{
        static_cast<unsigned char *>(base), requirement.bytes, 0u, requirement.where};
    return status::success;
}

status bind_launch_workspace(
    const runtime::launch_runtime_binding &binding,
    placement where,
    workspace *out) noexcept {
    if (out == nullptr) return status::invalid_argument;
    *out = workspace{};
    if (binding.status != runtime::session_status::success)
        return from_session_status(binding.status);
    if (where.kind != domain::device || where.device_ordinal < 0
        || where.numa_node != -1)
        return status::invalid_placement;
    if (binding.workspace_bytes != 0u && binding.workspace == nullptr)
        return status::invalid_state;
    *out = workspace{static_cast<unsigned char *>(binding.workspace),
        binding.workspace_bytes, 0u, where};
    return status::success;
}

} // namespace cellerator::memory
