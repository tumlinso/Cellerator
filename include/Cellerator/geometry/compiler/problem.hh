#pragma once

#include <Cellerator/execution/biological_abi.hh>
#include <Cellerator/geometry/work_window.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::geometry {

struct admissibility_view_v1;

namespace compiler {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 geometry_problem_schema_version = 1u;
inline constexpr u32 geometry_workload_profile_schema_version = 1u;

enum class geometry_operation_kind : u16 {
    relation_apply = 1u,
    relation_apply_transpose = 2u,
    contract_on_support = 3u,
    segment_operation = 4u,
    relation_bundle_apply = 5u
};

// This profile carries only operation properties that can change profitable
// semantic organization. It does not select a kernel or encode a device.
struct geometry_workload_profile_v1 {
    u32 schema_version = geometry_workload_profile_schema_version;
    geometry_operation_kind operation = geometry_operation_kind::relation_apply;
    u16 reserved = 0u;
    execution::numeric_type relation_value_type = execution::numeric_type::invalid;
    execution::numeric_type dense_input_type = execution::numeric_type::invalid;
    execution::numeric_type accumulation_type = execution::numeric_type::invalid;
    execution::numeric_type output_type = execution::numeric_type::invalid;
    u32 dense_width = 0u;
    u32 reserved2 = 0u;
    u64 expected_reuse = 1u;
};

// Optional evidence remains an opaque, versioned, source-linked data contract.
// A strategy may understand it; public compilation never trusts it to certify
// the resulting work layout or exact logical-edge cover.
struct portable_support_evidence_v1 {
    u64 evidence_identity = 0u;
    u32 schema_version = 0u;
    u32 evidence_kind = 0u;
    const void *data = nullptr;
    u64 data_bytes = 0u;
};

// The primary relation view supplies typed axes, immutable structure identity
// and epoch, logical edge count, and an optional source projection view. The
// work window and admissibility storage are caller-owned cold inputs.
struct geometry_problem_v1 {
    u32 schema_version = geometry_problem_schema_version;
    u32 reserved = 0u;
    execution::sparse_relation_view primary_relation{};
    work_window_view_v1 work_window{};
    const admissibility_view_v1 *admissibility = nullptr;
    geometry_workload_profile_v1 workload{};
    portable_support_evidence_v1 support_evidence{};
};

static_assert(std::is_trivially_copyable<geometry_workload_profile_v1>::value,
    "geometry workload profiles must remain pointer-copyable");
static_assert(std::is_trivially_copyable<portable_support_evidence_v1>::value,
    "portable support evidence views must remain pointer-copyable");
static_assert(std::is_trivially_copyable<geometry_problem_v1>::value,
    "geometry problems must remain pointer-copyable");

} // namespace compiler
} // namespace cellerator::geometry
