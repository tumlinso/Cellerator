#pragma once

#include <Cellerator/compiler/lto/implement_explicit_program_planning_authorization_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::lto::v1 {

enum class link_input_role_v1 : std::uint8_t { plain_cpp = 0, cellerator_cpu, cellerator_cuda };
struct link_driver_input_v1 { std::string path; link_input_role_v1 role = link_input_role_v1::plain_cpp; artifact_identity_v1 field{}; };
struct link_driver_request_v1 { std::string conventional_linker; std::vector<std::string> linker_options; std::vector<link_driver_input_v1> inputs; program_planning_authorization_v1 authorization{}; };
struct link_driver_plan_v1 { std::vector<artifact_identity_v1> program_ceir_fields; std::vector<std::string> replacement_objects; std::vector<std::string> final_linker_arguments; bool authorized_lto_ran = false; };
enum class link_driver_status_v1 : std::uint8_t { valid = 0, linker_missing, duplicate_input, authorization_missing, field_identity_missing };

[[nodiscard]] link_driver_status_v1 build_cellerator_link_driver_plan_v1(const link_driver_request_v1&, link_driver_plan_v1*) noexcept;

}  // namespace cellerator::compiler::lto::v1
