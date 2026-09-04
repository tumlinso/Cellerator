#pragma once

#include <Cellerator/compiler/ir/semantic/implement_domain_and_axis_ir_types_v1.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::ir::semantic {

enum class control_region_kind_ir_v1 : std::uint8_t {
    sequence = 1,
    branch,
    loop,
    opaque_cxx_control,
};

enum control_effect_ir_v1 : std::uint32_t {
    control_effect_none_v1 = 0,
    control_effect_reads_v1 = 1u << 0,
    control_effect_writes_v1 = 1u << 1,
    control_effect_synchronizes_v1 = 1u << 2,
    control_effect_opaque_barrier_v1 = 1u << 3,
};

struct profile_alternative_ir_v1 {
    semantic_identity_v1 profile{};
    double probability = 0.0;
};

struct control_value_state_ir_v1 {
    semantic_identity_v1 value{};
    std::uint64_t generation = 0;
    std::uint32_t effects = control_effect_none_v1;
};

struct control_dataflow_state_ir_v1 {
    std::vector<profile_alternative_ir_v1> profiles;
    std::vector<control_value_state_ir_v1> values;
    std::uint32_t effects = control_effect_none_v1;
};

struct control_region_ir_v1 {
    std::uint64_t identity = 0;
    control_region_kind_ir_v1 kind = control_region_kind_ir_v1::sequence;
    std::vector<std::uint64_t> child_regions;
    std::uint64_t bounded_trip_count = 0;
    std::uint32_t effects = control_effect_none_v1;
    bool semantic_extraction_available = true;
};

enum class control_flow_status_ir_v1 : std::uint8_t {
    success = 0,
    invalid_region,
    invalid_structure,
    invalid_profile,
    profile_alternative_limit,
    invalid_dataflow,
};

[[nodiscard]] control_flow_status_ir_v1
validate_control_regions_ir_v1(const std::vector<control_region_ir_v1>& regions) noexcept;

[[nodiscard]] control_flow_status_ir_v1
join_control_dataflow_ir_v1(
    const control_dataflow_state_ir_v1& left,
    const control_dataflow_state_ir_v1& right,
    std::size_t maximum_profile_alternatives,
    control_dataflow_state_ir_v1* result) noexcept;

}  // namespace Cellerator::compiler::ir::semantic
