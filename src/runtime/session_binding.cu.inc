launch_runtime_binding bind_launch(
    execution_session *session,
    std::uint32_t stream_index,
    std::size_t required_workspace_bytes) noexcept {
    launch_runtime_binding binding{};
    if (session == nullptr || !session->initialized || !session->sealed) return binding;
    if (stream_index >= session->stream_count) {
        binding.status = session_status::invalid_argument;
        return binding;
    }
    session_stream_slot &slot = session->streams[stream_index];
    if (slot.execution.device != session->device) {
        binding.status = session_status::device_mismatch;
        return binding;
    }
    if (!slot.libraries_prepared) return binding;
    if (required_workspace_bytes > slot.transient.bytes) {
        binding.status = session_status::workspace_exhausted;
        return binding;
    }
    binding.status = session_status::success;
    binding.execution = slot.execution;
    binding.cublas = slot.cublas.handle;
    binding.cusparse = slot.cusparse.handle;
    binding.workspace = slot.transient.data;
    binding.workspace_bytes = required_workspace_bytes;
    ++session->accounting.launch_bind_count;
    return binding;
}

bool graph_stable_address(
    const execution_session &session,
    const void *address,
    std::size_t bytes) noexcept {
    if (!session.initialized || !session.sealed || address == nullptr) return false;
    const std::uintptr_t query = reinterpret_cast<std::uintptr_t>(address);
    for (std::uint32_t index = 0;
         index < session.persistent_allocation_count; ++index) {
        const persistent_allocation_record &record =
            session.persistent_allocations[index];
        if (!record.occupied
            || record.lifetime != persistent_lifetime::graph_stable
            || record.data == nullptr || bytes > record.bytes) continue;
        const std::uintptr_t begin =
            reinterpret_cast<std::uintptr_t>(record.data);
        if (query >= begin && query - begin <= record.bytes - bytes) return true;
    }
    return false;
}

device_fleet_view single_device_fleet(const execution_session &session) noexcept {
    return {&session.performance, session.initialized ? 1u : 0u};
}
