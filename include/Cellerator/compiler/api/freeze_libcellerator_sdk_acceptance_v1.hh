#pragma once
#include <array>
#include <string_view>
namespace cellerator::compiler::api::v1 {
inline constexpr std::array<std::string_view,3> accepted_sdk_umbrellas_v1{"Cellerator/compiler/api/cellerator_compiler.h","Cellerator/compiler/api/compiler.hpp","Cellerator/sdk/runtime.hpp"};
[[nodiscard]] bool sdk_public_header_is_dependency_clean_v1(std::string_view contents) noexcept;
}
