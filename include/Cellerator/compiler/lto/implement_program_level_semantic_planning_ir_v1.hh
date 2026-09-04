#pragma once
#include <Cellerator/compiler/lto/freeze_the_ceir_companion_object_artifact_contract_v1.hh>
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::lto::v1 {
struct field_program_ir_v1{artifact_identity_v1 identity{};std::string source;bool exported=false;std::vector<artifact_identity_v1>calls,profiles,shared_artifacts;std::vector<std::string>semantic_nodes,external_effects,program_constraints;};
struct program_semantic_planning_ir_v1{std::vector<field_program_ir_v1>fields;std::vector<artifact_identity_v1>profiles,shared_artifacts;std::vector<std::string>external_effects,program_constraints,source_trace;};
enum class program_ir_merge_status_v1:std::uint8_t{valid=0,duplicate_field,missing_call_target,missing_source};
[[nodiscard]] program_ir_merge_status_v1 merge_program_semantic_planning_ir_v1(const std::vector<field_program_ir_v1>&,program_semantic_planning_ir_v1*)noexcept;
}
