#pragma once

#include <cstdint>
#include <string>

namespace Cellerator::compiler::sema::field {

enum class compilation_activation_kind_v1 : std::uint8_t {
    pure_cpp_fallthrough = 1,
    ceir_structural_only,
    biological_compilation,
};

struct missing_profile_policy_request_v1 {
    compilation_activation_kind_v1 activation =
        compilation_activation_kind_v1::pure_cpp_fallthrough;
    bool representative_profile_bound = false;
    bool generic_reference_profile_selected = false;
};

struct missing_profile_policy_result_v1 {
    bool compilation_allowed = false;
    bool uses_generic_reference_profile = false;
    std::string diagnostic;
};

enum class missing_profile_policy_status_v1 : std::uint8_t {
    success = 0,
    invalid_output,
    representative_profile_required,
};

[[nodiscard]] missing_profile_policy_status_v1 implement_missing_profile_failure_policy_v1(
    const missing_profile_policy_request_v1& request,
    missing_profile_policy_result_v1* result) noexcept;

}  // namespace Cellerator::compiler::sema::field
