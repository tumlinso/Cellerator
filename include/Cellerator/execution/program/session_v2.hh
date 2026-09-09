#pragma once
#include <Cellerator/execution/program/program_v2.h>
#include <Cellerator/runtime/session.cuh>
#include <Cellerator/runtime/relation_value_readiness.hh>
#include <array>
#include <atomic>

namespace cellerator::execution::program {
enum class instance_status : std::uint8_t {
    success, invalid_state, invalid_binding, busy, device_mismatch,
    insufficient_workspace, preflight_rejected, launch_failed, runtime_failure
};
// Borrows the sole execution_session and immutable program. One instance per
// session stream; no allocator, stream creation or alternate stage executor.
// Session, program, stage state and payloads must outlive checked close.
// All adapter host calls are nonblocking serialized; direct session mutation
// while attached is forbidden. Session scratch cannot be rebound after sealing.
class program_session_v2 {
public:
    program_session_v2() = default;
    ~program_session_v2();
    program_session_v2(const program_session_v2&) = delete;
    program_session_v2& operator=(const program_session_v2&) = delete;
    instance_status initialize(runtime::execution_session&, const prepared_program_v2&,
                               execution::structure_id, execution::structure_epoch) noexcept;
    instance_status binding(std::uint32_t instance, runtime::launch_runtime_binding&) noexcept;
    instance_status execute(std::uint32_t instance, const launch_binding_v2*,
                            std::uint64_t count, execution::value_generation next) noexcept;
    instance_status begin_read(std::uint32_t instance, execution::value_generation,
                               cudaStream_t consumer, runtime::relation_read_ticket&) noexcept;
    instance_status end_read(std::uint32_t instance, runtime::relation_read_ticket&,
                             cudaStream_t consumer) noexcept;
    instance_status close() noexcept;
private:
    struct guard {
        std::atomic_flag& flag;
        bool acquired;
        explicit guard(std::atomic_flag& f) : flag(f), acquired(!f.test_and_set()) {}
        ~guard() { if (acquired) flag.clear(); }
    };
    instance_status context(std::uint32_t) const noexcept;
    std::atomic_flag active_ = ATOMIC_FLAG_INIT;
    runtime::execution_session* session_ = nullptr;
    const prepared_program_v2* program_ = nullptr;
    execution::structure_id identity_{};
    execution::structure_epoch epoch_{};
    std::uint64_t workspace_bytes_ = 0;
    std::uint32_t instance_count_ = 0;
    std::array<runtime::launch_runtime_binding,runtime::execution_session_max_streams> bindings_{};
    std::array<runtime::relation_value_readiness,runtime::execution_session_max_streams> readiness_{};
};
} // namespace cellerator::execution::program
