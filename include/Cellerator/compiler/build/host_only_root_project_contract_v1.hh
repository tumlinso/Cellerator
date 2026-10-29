#pragma once

#include <array>
#include <string_view>

namespace Cellerator::compiler::build {

enum class accelerator_enablement_v1 {
    automatic,
    enabled,
    disabled,
};

struct root_project_contract_v1 {
    std::array<std::string_view, 1> required_languages;
    accelerator_enablement_v1 default_accelerator_enablement;
    bool accelerator_language_is_optional;
};

inline constexpr root_project_contract_v1 host_only_root_project_contract_v1{
    {"CXX"},
    accelerator_enablement_v1::automatic,
    true,
};

[[nodiscard]] constexpr bool is_host_only(
    const root_project_contract_v1& contract) noexcept {
    return contract.required_languages.size() == 1 &&
           contract.required_languages.front() == "CXX" &&
           contract.accelerator_language_is_optional;
}

}  // namespace Cellerator::compiler::build
