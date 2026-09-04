#pragma once
#include <cstddef>
#include <cstdint>
namespace cellerator::compiler::api::v1 {
struct runtime_axis_v1 { std::uint64_t domain=0, order=0, extent=0; };
struct runtime_relation_v1 { runtime_axis_v1 source{}, destination{}; std::uint64_t structure=0, epoch=0, edges=0; };
struct runtime_value_plane_v1 { const void* data=nullptr; std::size_t bytes=0; std::uint64_t generation=0; };
struct runtime_launch_v1 { void* stream=nullptr; void* workspace=nullptr; std::size_t workspace_bytes=0; };
enum class runtime_status_v1 : std::uint8_t { ok, invalid_identity, stale_generation, insufficient_workspace, backend_failure };
using runtime_execute_v1=runtime_status_v1(*)(const runtime_relation_v1&,const runtime_value_plane_v1&,const runtime_launch_v1&,void*) noexcept;
struct runtime_facade_v1 { runtime_execute_v1 execute=nullptr; void* user_data=nullptr; };
[[nodiscard]] runtime_status_v1 execute_v1(const runtime_facade_v1&,const runtime_relation_v1&,const runtime_value_plane_v1&,const runtime_launch_v1&) noexcept;
}
