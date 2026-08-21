namespace {

allocation_accounting *accounting_for(
    execution_session *session,
    persistent_lifetime lifetime) noexcept {
    if (lifetime == persistent_lifetime::structure) return &session->accounting.structure;
    if (lifetime == persistent_lifetime::plan) return &session->accounting.plan;
    return &session->accounting.graph_stable;
}

fixed_session_cache *cache_for(
    execution_session *session,
    session_cache_kind kind) noexcept {
    if (kind == session_cache_kind::plan) return &session->plans;
    if (kind == session_cache_kind::projection) return &session->projections;
    return &session->order_transforms;
}

const fixed_session_cache *cache_for(
    const execution_session &session,
    session_cache_kind kind) noexcept {
    if (kind == session_cache_kind::plan) return &session.plans;
    if (kind == session_cache_kind::projection) return &session.projections;
    return &session.order_transforms;
}

bool same_key(session_cache_key left, session_cache_key right) noexcept {
    return left.high == right.high && left.low == right.low;
}

bool valid_persistent_lifetime(persistent_lifetime lifetime) noexcept {
    return lifetime == persistent_lifetime::structure
        || lifetime == persistent_lifetime::plan
        || lifetime == persistent_lifetime::graph_stable;
}

} // namespace

session_status init_session(
    execution_session *session,
    const execution_session_options &options) noexcept {
    if (session == nullptr) return session_status::invalid_argument;
    if (options.external_stream_count != 0 && options.external_streams == nullptr) {
        return session_status::invalid_argument;
    }
    const std::uint32_t stream_count =
        options.external_stream_count + options.owned_stream_count;
    if (stream_count == 0 || stream_count > execution_session_max_streams) {
        return session_status::capacity_exceeded;
    }

    *session = execution_session{};
    try {
        session->device = options.device;
        if (session->device < 0) {
            cuda_require(cudaGetDevice(&session->device), "cudaGetDevice(execution session)");
            ++session->accounting.device_query_count;
        }
        cudaDeviceProp properties{};
        cuda_require(
            cudaGetDeviceProperties(&properties, session->device),
            "cudaGetDeviceProperties(execution session)");
        ++session->accounting.device_query_count;
        session->performance.device = session->device;
        session->performance.compute_major = properties.major;
        session->performance.compute_minor = properties.minor;
        session->performance.multiprocessor_count = properties.multiProcessorCount;
        session->performance.warp_size = properties.warpSize;
        session->performance.global_memory_bytes = properties.totalGlobalMem;

        for (std::uint32_t index = 0; index < stream_count; ++index) {
            session_stream_slot &slot = session->streams[index];
            init(&slot.cublas);
            init(&slot.cusparse);
            init(&slot.transient);
            const cudaStream_t external = index < options.external_stream_count
                ? options.external_streams[index]
                : nullptr;
            init(&slot.execution, session->device, external);
        }
        session->stream_count = stream_count;
        session->initialized = true;
        return session_status::success;
    } catch (...) {
        clear_session(session);
        return session_status::cuda_failure;
    }
}

void clear_session(execution_session *session) noexcept {
    if (session == nullptr) return;
    if (session->initialized && session->device >= 0) (void) cudaSetDevice(session->device);
    for (std::uint32_t index = 0; index < execution_session_max_streams; ++index) {
        session_stream_slot &slot = session->streams[index];
        clear(&slot.cusparse);
        clear(&slot.cublas);
        clear(&slot.transient);
        clear(&slot.execution);
        slot.libraries_prepared = false;
    }
    for (std::uint32_t index = 0;
         index < execution_session_max_persistent_allocations; ++index) {
        persistent_allocation_record &record =
            session->persistent_allocations[index];
        if (record.occupied && record.data != nullptr) (void) cudaFree(record.data);
        record = persistent_allocation_record{};
    }
    *session = execution_session{};
}

session_status reserve_persistent(
    execution_session *session,
    persistent_lifetime lifetime,
    std::size_t bytes,
    void **allocation) noexcept {
    if (allocation != nullptr) *allocation = nullptr;
    if (session == nullptr || allocation == nullptr) return session_status::invalid_argument;
    if (!session->initialized || session->sealed) return session_status::invalid_state;
    if (!valid_persistent_lifetime(lifetime)) return session_status::invalid_argument;
    if (bytes == 0u) return session_status::success;
    if (session->persistent_allocation_count
        == execution_session_max_persistent_allocations) {
        return session_status::capacity_exceeded;
    }
    if (cudaSetDevice(session->device) != cudaSuccess) {
        return session_status::cuda_failure;
    }
    void *data = nullptr;
    if (cudaMalloc(&data, bytes) != cudaSuccess) return session_status::cuda_failure;
    persistent_allocation_record &record = session->persistent_allocations[
        session->persistent_allocation_count++];
    record = persistent_allocation_record{data, bytes, lifetime, true};
    allocation_accounting *accounting = accounting_for(session, lifetime);
    accounting->current_bytes += bytes;
    if (accounting->current_bytes > accounting->high_water_bytes)
        accounting->high_water_bytes = accounting->current_bytes;
    ++accounting->allocation_count;
    *allocation = data;
    return session_status::success;
}

session_status reserve_transient(
    execution_session *session,
    std::uint32_t stream_index,
    std::size_t bytes,
    void **allocation) noexcept {
    if (allocation != nullptr) *allocation = nullptr;
    if (session == nullptr || allocation == nullptr) return session_status::invalid_argument;
    if (!session->initialized || session->sealed) return session_status::invalid_state;
    if (stream_index >= session->stream_count) return session_status::invalid_argument;
    scratch_arena &arena = session->streams[stream_index].transient;
    const std::size_t previous_bytes = arena.bytes;
    try {
        *allocation = request_scratch(&arena, bytes);
    } catch (...) {
        return session_status::cuda_failure;
    }
    session->accounting.transient.current_bytes -= previous_bytes;
    session->accounting.transient.current_bytes += arena.bytes;
    if (session->accounting.transient.current_bytes
        > session->accounting.transient.high_water_bytes) {
        session->accounting.transient.high_water_bytes =
            session->accounting.transient.current_bytes;
    }
    if (arena.bytes > previous_bytes) ++session->accounting.transient.allocation_count;
    return session_status::success;
}

session_status prepare_stream_libraries(
    execution_session *session,
    std::uint32_t stream_index) noexcept {
    if (session == nullptr || !session->initialized || session->sealed) {
        return session_status::invalid_state;
    }
    if (stream_index >= session->stream_count) return session_status::invalid_argument;
    session_stream_slot &slot = session->streams[stream_index];
    try {
        acquire_cublas(&slot.cublas, slot.execution);
        acquire_cusparse(&slot.cusparse, slot.execution);
    } catch (...) {
        return session_status::cuda_failure;
    }
    slot.libraries_prepared = true;
    ++session->accounting.handle_prepare_count;
    return session_status::success;
}

session_status seal_session(execution_session *session) noexcept {
    if (session == nullptr || !session->initialized || session->sealed) {
        return session_status::invalid_state;
    }
    for (std::uint32_t index = 0; index < session->stream_count; ++index) {
        if (!session->streams[index].libraries_prepared) return session_status::invalid_state;
    }
    session->sealed = true;
    return session_status::success;
}
