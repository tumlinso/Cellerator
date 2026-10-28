#pragma once
#include <array>
#include <string>
#include <string_view>
namespace cellerator::compiler::driver {
enum class override_source_v1 { command_line, response_file, environment, build_configuration, resource_manifest, system_discovery, unresolved };
struct override_candidates_v1 { std::array<std::string, 6> values{}; };
struct resolved_override_v1 { override_source_v1 source = override_source_v1::unresolved; std::string value; explicit operator bool() const noexcept { return source != override_source_v1::unresolved; } };
resolved_override_v1 resolve_toolchain_override_v1(const override_candidates_v1&) noexcept;
std::string_view override_source_name_v1(override_source_v1) noexcept;
}  // namespace cellerator::compiler::driver
