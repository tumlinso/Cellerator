#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::lto::v1 {
enum class emission_backend_v1:std::uint8_t{conventional=0,cellerator_cpu,cellerator_cuda,native};
enum class emission_action_v1:std::uint8_t{retain=0,reemit};
struct backend_artifact_v1{std::string input,output;emission_backend_v1 backend=emission_backend_v1::conventional;bool program_region_changed=false,artifact_valid=true;};
struct backend_emission_v1{std::string input,output;emission_backend_v1 backend=emission_backend_v1::conventional;emission_action_v1 action=emission_action_v1::retain;};
[[nodiscard]] std::vector<backend_emission_v1> plan_mixed_backend_re_emission_v1(const std::vector<backend_artifact_v1>&);
}
