#pragma once
#include <array>
#include <string_view>
namespace cellerator::compiler::api::v1 {
inline constexpr std::array<std::string_view,6> installed_sdk_examples_v1{"runtime","source-compiler","ceir-editing","custom-candidate","custom-pass","backend"};
[[nodiscard]] bool has_installed_sdk_example_v1(std::string_view) noexcept;
}
