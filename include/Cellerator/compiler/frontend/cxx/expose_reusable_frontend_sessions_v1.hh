#pragma once

#include <Cellerator/compiler/frontend/cxx/create_the_c_compilation_invocation_bridge_v1.hh>
#include <Cellerator/compiler/frontend/cxx/freeze_the_upstream_clang_adapter_boundary_v1.hh>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {

inline constexpr std::uint32_t reusable_frontend_session_schema_version_v1 = 1;

enum class reusable_frontend_session_status_v1 : std::uint8_t {
    success = 0,
    null_output,
    schema_mismatch,
    unsupported_llvm_major,
    missing_invocation,
    empty_source,
    cancelled,
    clang_parse_failed,
    semantic_errors,
};

class frontend_cancellation_token_v1 {
public:
    frontend_cancellation_token_v1() noexcept;
    void cancel() noexcept;
    bool is_cancelled() const noexcept;

private:
    struct state;
    std::shared_ptr<state> state_;
};

struct reusable_frontend_parse_request_v1 {
    std::uint32_t schema_version = reusable_frontend_session_schema_version_v1;
    const cxx_compilation_invocation_v1* invocation = nullptr;
    std::string virtual_filename = "cellerator-session.cc";
    std::string source;
    frontend_cancellation_token_v1 cancellation;
};

class immutable_frontend_snapshot_v1 {
public:
    immutable_frontend_snapshot_v1() noexcept;
    ~immutable_frontend_snapshot_v1();
    immutable_frontend_snapshot_v1(const immutable_frontend_snapshot_v1&) noexcept;
    immutable_frontend_snapshot_v1& operator=(const immutable_frontend_snapshot_v1&) noexcept;
    immutable_frontend_snapshot_v1(immutable_frontend_snapshot_v1&&) noexcept;
    immutable_frontend_snapshot_v1& operator=(immutable_frontend_snapshot_v1&&) noexcept;

    bool valid() const noexcept;
    std::uint64_t sequence() const noexcept;
    std::uint64_t worker_thread_id() const noexcept;
    std::string_view virtual_filename() const noexcept;
    const std::vector<std::string>& errors() const noexcept;
    const std::vector<std::string>& warnings() const noexcept;
    const upstream_clang_adapter_v1& adapter() const noexcept;

private:
    struct implementation;
    std::shared_ptr<const implementation> implementation_;

    friend class reusable_frontend_session_v1;
};

class reusable_frontend_session_v1 {
public:
    explicit reusable_frontend_session_v1(
        std::uint32_t llvm_major = 18) noexcept;
    ~reusable_frontend_session_v1();
    reusable_frontend_session_v1(const reusable_frontend_session_v1&) noexcept;
    reusable_frontend_session_v1& operator=(const reusable_frontend_session_v1&) noexcept;
    reusable_frontend_session_v1(reusable_frontend_session_v1&&) noexcept;
    reusable_frontend_session_v1& operator=(reusable_frontend_session_v1&&) noexcept;

    std::uint32_t llvm_major() const noexcept;
    std::uint64_t completed_parse_count() const noexcept;
    reusable_frontend_session_status_v1 parse(
        const reusable_frontend_parse_request_v1& request,
        immutable_frontend_snapshot_v1* snapshot) const noexcept;

private:
    struct implementation;
    std::shared_ptr<implementation> implementation_;
};

}  // namespace Cellerator::compiler::frontend::cxx
