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

session_status insert_session_cache(
    execution_session *session,
    session_cache_kind kind,
    session_cache_key key,
    void *state,
    std::uint64_t structure_epoch,
    std::uint64_t generation) noexcept {
    if (session == nullptr || !session->initialized || session->sealed || state == nullptr) {
        return session_status::invalid_state;
    }
    fixed_session_cache *cache = cache_for(session, kind);
    for (std::uint32_t index = 0; index < cache->size; ++index) {
        if (same_key(cache->entries[index].key, key)) {
            cache->entries[index] = {key, state, structure_epoch, generation, true};
            return session_status::success;
        }
    }
    if (cache->size == execution_session_cache_capacity) {
        return session_status::capacity_exceeded;
    }
    cache->entries[cache->size++] = {key, state, structure_epoch, generation, true};
    return session_status::success;
}

const session_cache_entry *find_session_cache(
    const execution_session &session,
    session_cache_kind kind,
    session_cache_key key) noexcept {
    const fixed_session_cache *cache = cache_for(session, kind);
    for (std::uint32_t index = 0; index < cache->size; ++index) {
        if (cache->entries[index].occupied && same_key(cache->entries[index].key, key)) {
            return &cache->entries[index];
        }
    }
    return nullptr;
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
