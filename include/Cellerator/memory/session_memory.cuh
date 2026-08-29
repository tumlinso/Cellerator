#pragma once

#include "allocation.hh"
#include "workspace.hh"

#include <Cellerator/runtime/session.cuh>

#include <cstddef>
#include <cstdint>

namespace cellerator::memory {

status reserve_session_allocation(
    runtime::execution_session *session,
    runtime::persistent_lifetime lifetime,
    const allocation_request &request,
    std::uint32_t generation,
    allocation *out) noexcept;

status reserve_session_workspace(
    runtime::execution_session *session,
    std::uint32_t stream_index,
    const workspace_requirement &requirement,
    workspace *out) noexcept;

status bind_launch_workspace(
    const runtime::launch_runtime_binding &binding,
    placement where,
    workspace *out) noexcept;

} // namespace cellerator::memory
