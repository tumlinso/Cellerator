#pragma once

#include <Cellerator/execution/geometry_acquisition_v2/projections.hh>
#include <Cellerator/execution/joint_compiler/atom_fragment_request_v1.hh>
#include <Cellerator/execution/program/program_v2.h>
#include <Cellerator/planner/portfolio/candidate_workspace_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::atom_fragment {

inline constexpr std::uint32_t atom_fragment_requirements_schema_version_v1 = 1u;

struct atom_fragment_buffer_requirement_v1 {
    std::uint64_t bytes = 0u;
    std::uint64_t alignment = 1u;
};

struct atom_fragment_query_limits_v1 {
    std::uint64_t projection_capacity = 0u;
    std::uint64_t projection_chunk_capacity = 0u;
    std::uint64_t candidate_capacity = 0u;
    std::uint64_t prepared_stage_capacity = 0u;
    std::uint64_t dependency_capacity = 0u;
    std::uint64_t diagnostic_capacity = 0u;
    std::uint64_t transient_bytes = 0u;
};

struct atom_fragment_diagnostic_record_v1 {
    std::uint64_t subject = 0u;
    std::uint32_t code = 0u;
    std::uint32_t detail = 0u;
};

struct atom_fragment_requirements_v1 {
    std::uint32_t schema_version = atom_fragment_requirements_schema_version_v1;
    std::uint32_t record_bytes = sizeof(atom_fragment_requirements_v1);
    std::uint64_t local_index_component_count = 0u;
    std::uint64_t projection_capacity = 0u;
    std::uint64_t projection_chunk_capacity = 0u;
    std::uint64_t candidate_capacity = 0u;
    std::uint64_t prepared_stage_capacity = 0u;
    std::uint64_t dependency_capacity = 0u;
    std::uint64_t binding_capacity = 0u;
    std::uint64_t diagnostic_capacity = 0u;
    atom_fragment_buffer_requirement_v1 local_indexes{};
    atom_fragment_buffer_requirement_v1 projections{};
    atom_fragment_buffer_requirement_v1 candidate_workspace{};
    atom_fragment_buffer_requirement_v1 prepared_program{};
    atom_fragment_buffer_requirement_v1 bindings{};
    atom_fragment_buffer_requirement_v1 diagnostics{};
    atom_fragment_buffer_requirement_v1 transient{};
};

enum class atom_fragment_requirements_status_code_v1 : std::uint8_t {
    success = 0u,
    null_output,
    invalid_request,
    invalid_limits,
    arithmetic_overflow,
};

struct atom_fragment_requirements_status_v1 {
    atom_fragment_requirements_status_code_v1 code =
        atom_fragment_requirements_status_code_v1::success;
    std::uint64_t subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == atom_fragment_requirements_status_code_v1::success;
    }
};

atom_fragment_requirements_status_v1 query_atom_fragment_requirements_v1(
    const joint_compiler::atom_fragment_request_v1 &request,
    const atom_fragment_query_limits_v1 &limits,
    atom_fragment_requirements_v1 *requirements) noexcept;

static_assert(std::is_standard_layout_v<atom_fragment_query_limits_v1>);
static_assert(std::is_trivially_copyable_v<atom_fragment_query_limits_v1>);
static_assert(std::is_standard_layout_v<atom_fragment_requirements_v1>);
static_assert(std::is_trivially_copyable_v<atom_fragment_requirements_v1>);

} // namespace cellerator::execution::atom_fragment
